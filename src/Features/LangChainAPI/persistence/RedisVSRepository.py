from collections import defaultdict
import json
import logging
from typing import Any, Dict, List
from langchain_community.storage.redis import RedisStore
from redis import Redis
from redisvl.index import SearchIndex
from langchain_core.documents import Document
from redisvl.query import FilterQuery
from SharedKernel.ai.AIConfig import AIConfigFactory
from SharedKernel.ai.vector_store.VectorStoreConfig import VectoreStoreConfigFactory
from SharedKernel.utils.yamlenv import load_env_yaml
from src.Features.LangChainAPI.RAG.Retriever import HybridRetriever

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
config = load_env_yaml()

class RedisVSRepository:
    """
    Repository for Redis Vector Store data access operations.
    Handles CRUD operations for documents in vector store.
    """

    def __init__(self, ai_factory: AIConfigFactory):
        self.ai_config = ai_factory.create(config.ai.llm_provider)
        self.embeddings = self.ai_config.create_embedding()
        self.vs_config = VectoreStoreConfigFactory.create(config.vector_store.provider)
        self.redis_vs = self.vs_config.get_vecstore(self.embeddings)
        self.redis_url = self.vs_config.get_url()
        self.store = RedisStore(redis_url=self.redis_url)

    # ============================================================
    # DOCUMENT OPERATIONS (CRUD)
    # ============================================================

    async def abatch_add_documents(self, documents: List[Any]):
        """Add documents in batches to vector store"""
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

    async def add_documents_with_metadata(self, documents: List[Document]):
        """Add documents with metadata to vector store"""
        if documents:
            print(f"Adding {len(documents)} documents with metadata...")
            await self.redis_vs.aadd_documents(documents)
            print(f"Successfully added {len(documents)} documents")

            sources = set(doc.metadata.get('source', 'unknown') for doc in documents)
            print(f"Sources: {list(sources)}")
        else:
            print("No documents to add")

    async def abatch_add_documents_with_metadata(self, chunks: List[Document]):
        """Batch add documents with enhanced metadata"""
        import time

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
        """Add Parent-Child documents to vector store"""
        parent_ids = []
        for parent in chunks.get("parent", []):
            key = parent.metadata.get('parent_id', "")
            self.store.mset([(key, parent.page_content)])
            parent_ids.append(key)
            print(f"Saved parent: {key}")

        print(chunks.get('children'))
        await self.abatch_add_documents_with_metadata(chunks.get('children', []))
        print(f"Saved child")

    async def add_PaC_documents_v2(self, chunks: Any):
        """Add Parent-Child documents to vector store with parent metadata"""
        parent_ids = []
        for parent in chunks.get("parent", []):
            key = parent.metadata.get('parent_id', "")
            parent_data = {
                "page_content": parent.page_content,
                "metadata": parent.metadata
            }
            self.store.mset([(key, json.dumps(parent_data))])
            parent_ids.append(key)
            print(f"Saved parent with metadata: {key}")

        print(chunks.get('children'))
        await self.abatch_add_documents_with_metadata(chunks.get('children', []))
        print(f"Saved child")

    async def delete_documents_by_metadata(self, metadata: dict):
        """Delete documents from vector store by metadata"""
        try:
            r = Redis.from_url(self.redis_url)
            index = SearchIndex.from_yaml("config/redis_index.yaml")
            index.connect(redis_url=self.redis_url)

            target_source = metadata["source"]
            filter_expression = f'@source:{{"{target_source}"}}'

            BATCH_SIZE = 500
            PIPELINE_BATCH = 500

            offset = 0
            deleted_count = 0

            while True:
                filter_query = FilterQuery(
                    return_fields=["id"],
                    filter_expression=filter_expression,
                    num_results=BATCH_SIZE
                )

                results = index.query(filter_query)
                if not results:
                    break

                pipeline = r.pipeline()
                batch_count = 0

                for result in results:
                    doc_id = result.get("id")
                    if doc_id:
                        pipeline.unlink(doc_id)
                        batch_count += 1
                        deleted_count += 1

                    if batch_count == PIPELINE_BATCH:
                        pipeline.execute()
                        pipeline = r.pipeline()
                        batch_count = 0

                pipeline.execute()
                offset += BATCH_SIZE
                print(f"Deleted batch, total: {deleted_count}")

            print(f"Deleted {deleted_count} vector documents for source: {target_source}")

            # Delete parent docs
            cursor = 0
            deleted_parents = 0
            while True:
                cursor, keys = r.scan(cursor, match=f"parent_docs:{target_source}:*", count=100)
                if not keys:
                    if cursor == 0:
                        break
                    continue

                pipeline = r.pipeline()
                batch_count = 0

                for key in keys:
                    key_str = key.decode('utf-8')
                    if target_source in key_str:
                        pipeline.unlink(key)
                        batch_count += 1
                        deleted_parents += 1

                    if batch_count == PIPELINE_BATCH:
                        pipeline.execute()
                        pipeline = r.pipeline()
                        batch_count = 0

                pipeline.execute()

                if cursor == 0:
                    break

            print(f"Deleted {deleted_count} vector docs and {deleted_parents} parent docs for source: {target_source}")
        except Exception as e:
            log.error(f"Error deleting documents: {e}")
            return

    # ============================================================
    # SEARCH/RETRIEVE OPERATIONS
    # ============================================================
    async def hybrid_retriver(self, query: str, k: int = 5) -> List[Dict]:
        hre = HybridRetriever(self.embeddings, self.redis_url)
        return await hre.retriever(query, k)

    async def hybrid_retriver_v2(self, query: str, k: int = 5) -> List[Dict]:
        hre = HybridRetriever(self.embeddings, self.redis_url)
        return await hre.retriever_v2(query, k)
