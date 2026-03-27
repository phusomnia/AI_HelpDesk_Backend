import asyncio
from collections import defaultdict
import json
import logging
from re import search
import time
from typing import Any, Dict, List, Optional
from langchain_community.storage.redis import RedisStore
from langchain_core.stores import InMemoryStore
from langchain_redis import RedisVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from redis import Redis
from redisvl.index import SearchIndex
from redisvl.query import AggregateHybridQuery, FilterQuery, HybridQuery, TextQuery, VectorQuery, hybrid
from transformers.utils import doc
from SharedKernel.ai.AIConfig import AIConfigFactory
from SharedKernel.ai.vector_store.VectorStoreConfig import VectoreStoreConfigFactory
from SharedKernel.persistence.Decorators import Repository
from SharedKernel.utils.yamlenv import load_env_yaml
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever, MultiVectorRetriever, ParentDocumentRetriever

from src.Features.LangChainAPI.prompt import PaC_template

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
config = load_env_yaml()

class RedisVS:
    def __init__(self, ai_factory: AIConfigFactory):
        self.ai_config = ai_factory.create(config.ai.llm_provider)
        self.embeddings = self.ai_config.create_embedding()
        self.vs_config = VectoreStoreConfigFactory.create(config.vector_store.provider)
        self.redis_vs = self.vs_config.get_vecstore(self.embeddings)
        self.redis_url = self.vs_config.get_url()
        self.store = RedisStore(redis_url=self.redis_url)
        ...
    # 
    # DOCUMENT 
    # 
    async def abatch_add_documents(self, documents: List[Any]):
        if not documents:
            print("No documents to add")
            return []

        total_docs = len(documents)
        batch_size = min(100, total_docs) 

        all_ids = []
        for i in range(0, total_docs, batch_size):
            batch = documents[i:i + batch_size]
            batch_ids = await self.redis_vs.aadd_documents(batch)
            all_ids.extend(batch_ids)
            
            print(f"Processed batch {i//batch_size + 1}/{(total_docs - 1)//batch_size + 1}")

        print(f"Added {len(documents)} documents with metadata")
        return all_ids
        ...

    async def add_documents_with_metadata(self, documents: List[Document]):      
        if documents:
            print(f"Adding {len(documents)} documents with metadata...")
            
            await self.redis_vs.aadd_documents(documents)
            
            print(f"Successfully added {len(documents)} documents")
            
            sources = set(doc.metadata.get('source', 'unknown') for doc in documents)
            print(f"Sources: {list(sources)}")
        else:
            print("No documents to add")

    async def abatch_add_documents_with_metadata(self, chunks: List[Document]):
        documents = []
        current_time = int(time.time())

        for i, chunk in enumerate(chunks):
            metadata = chunk.metadata if chunk.metadata else {}

            metadata.update({
                "chunk_index": i,
                "total_chunks": len(chunks),
                "timestamp": current_time,
                "content_length": len(chunk.page_content)
            })
        
            documents.append(Document(
                page_content=chunk.page_content,
                metadata=metadata
            ))

        print(f"Saving {len(documents)} documents to vector store")
    
        if documents:
            print(f"Sample metadata: {documents[0].metadata}")
        
        await self.abatch_add_documents(documents)

    async def add_PaC_documents(self, chunks: Any):
        parent_ids = []
        for parent in chunks.get("parent", []): 
            key = parent.metadata.get('parent_id', "")
            self.store.mset([(key, parent.page_content)])
            parent_ids.append(key)
            print(f"Saved parent: {key}")

        print(chunks.get('children'))
        await self.abatch_add_documents_with_metadata(chunks.get('children', []))
        print(f"Saved child")
        ...

    async def delete_documents_by_metadata(self, metadata: dict):
        r = Redis.from_url(self.redis_url)

        try:
            index = SearchIndex.from_yaml("config/redis_index.yaml")
            index.connect(redis_url=self.redis_url)

            target_source = metadata["source"]
            filter_expression = f'@source:{{"{target_source}"}}'
            filter_query = FilterQuery(
                return_fields=["id"],
                filter_expression=filter_expression,
                num_results=100000
            )

            results = index.query(filter_query)
            if results:
                print(f"Found documents to delete: {len(results)}")
            else:
                print("No documents found to delete")

            deleted_count = 0
            for result in results:
                doc_id = result.get("id")
                if doc_id:
                    r.delete(doc_id)
                    deleted_count += 1
                    print(f"Deleted vector doc: {doc_id}")
            print(f"Deleted {deleted_count} documents")

            cursor = 0
            deleted_parents = 0
            while True:
                cursor, keys = r.scan(cursor, match=f"parent_docs:{target_source}:*", count=100)
                print(f"Cursor: {cursor}")
                if keys:
                    for key in keys:
                        key_str = key.decode('utf-8')
                        if target_source in key_str:
                            r.delete(key)
                            deleted_parents += 1
                            print(f"Deleted parent: {key_str}")
                if cursor == 0:
                    break
            
            print(f"Deleted {deleted_count} vector docs and {deleted_parents} parent docs for source: {target_source}")
        except Exception as e:
            log.error(f"Error deleting documents: {e}")
            return
        ...

    async def rag_PaC(self,         
        query: str,
        provider
    ):
        hybrid_retriver = HybridRetriever(self.embeddings, self.redis_url)
        hybrid_docs = await hybrid_retriver.asearch(query)
        context = self._format_context_PaC(hybrid_docs)

        template = PaC_template(context, query)
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | provider 
        answer = await chain.ainvoke({"query": query})
        return answer
    ...

    def _format_context_PaC(self,         
        search_results: List[Dict[str, Any]]):

        if not search_results:
            return "No relevant documents found."

        context_parts = []
        for idx, result in enumerate(search_results):
            doc_content = result["content"].replace("\n", " ").strip()
            file_name = result["id"].split(":")[1]
            page = result["id"].split(":")[2].split("_")[1]
            
            metadata_info = []

            if file_name:
                metadata_info.append(f"Source: {file_name}")
            if page:
                metadata_info.append(f"Page: {page}")
            
            doc_content = doc_content + "\n" + (" | ".join(metadata_info))
            context_parts.append(doc_content)

        formatted_context = "\n\n".join(context_parts)
        return formatted_context
    ...
    
class HybridRetriever:
    def __init__(self, embeddimgs, redis_url):
        self.embeddings = embeddimgs
        self.redis_url = redis_url
        self.index = SearchIndex.from_yaml("config/redis_index.yaml")
        self.index.connect(redis_url=self.redis_url)
        self.store = RedisStore(redis_url=self.redis_url)
        ...
    
    async def asearch(self, query: str, k: int = 5, threshold: float = 0.0):
        query_embed = await self.embeddings.aembed_query(query)

        vector_query = VectorQuery(
            vector=query_embed,
            vector_field_name="embedding",
            num_results=k,
            return_fields=["_metadata_json", "text"]
        )

        bm25_query = TextQuery(
            text=query,
            text_field_name="text",
            num_results=k,
            return_fields=["_metadata_json", "text"]
        )

        vector_docs = self.index.query(vector_query)
        bm25_docs = self.index.query(bm25_query)
        print(bm25_docs)

        fused = self.rrf_fusion([bm25_docs, vector_docs])

        filtered_score_fused = []
        for doc_id, score in fused:
            if score >= threshold:
                filtered_score_fused.append((doc_id, score))

        top_docs = filtered_score_fused[:k]
        print(top_docs)

        doc_map = {}
        for doc in list(vector_docs) + list(bm25_docs):
            metadata = json.loads(doc["_metadata_json"])

            doc_map[doc["id"]] = {
                "text": doc["text"],
                "metadata": metadata
            }

        parent_to_children = defaultdict(list)
        for doc_id, _ in top_docs:
            metadata = doc_map[doc_id]["metadata"].copy()
            metadata.pop('parent_id', None) 
            parent_id = doc_map[doc_id]["metadata"]["parent_id"]
            parent_to_children[parent_id].append({
                "id": doc_id, "metadata": metadata
            })

        parent_ids = list(parent_to_children.keys())
        parent_docs = self.store.mget(parent_ids)

        results = []
        for i, parent in enumerate(parent_docs):

            if not parent:
                continue

            parent_id = parent_ids[i]
            parent_text = parent.decode()

            child_ids = parent_to_children[parent_id]

            results.append({
                "id": parent_id,
                "content": parent_text,
                "children": child_ids,
            })
        
        return results

    def rrf_fusion(self, rank_lists, k: int = 60):
        """
        Reciprocal Rank Fusion
        """
        print("RRF...")

        score_map = defaultdict(float)
        for ranking in rank_lists:
            for rank, doc in enumerate(ranking, start=1):
                doc_id = doc["id"]
                score_map[doc_id] += 1 / (k + rank)
        return sorted(score_map.items(), key=lambda x: x[1], reverse=True)
    