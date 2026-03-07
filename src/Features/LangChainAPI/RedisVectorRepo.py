import logging
from re import search
import time
from typing import Any, Dict, List, Optional
from SharedKernel.ai.AIConfig import AIConfig, AIConfigFactory
from SharedKernel.ai.vector_store.VectorStoreConfig import VectoreStoreConfigFactory
from SharedKernel.persistence.Decorators import Repository
from SharedKernel.utils.yamlenv import load_env_yaml
from fastapi import UploadFile
from langchain_core.documents import Document
from langchain_core.vectorstores import VectorStore
from langchain_core.retrievers import BaseRetriever
from langchain_core.prompts import ChatPromptTemplate

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
config = load_env_yaml()

class RedisVectorRepo:
    def __init__(self, ai_factory: AIConfigFactory):
        self.ai_config = ai_factory.create(config.ai.llm_provider)
        self.embeddings = self.ai_config.create_embedding()
        self.vector_store_config = VectoreStoreConfigFactory.create(config.vector_store.provider)
        self.vs_repo = self.vector_store_config.get_vecstore(self.embeddings)
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
            batch_ids = await self.vs_repo.aadd_documents(batch)
            all_ids.extend(batch_ids)
            
            print(f"Processed batch {i//batch_size + 1}/{(total_docs - 1)//batch_size + 1}")

        print(f"Added {len(documents)} documents with metadata")
        return all_ids
        ...

    async def add_documents_with_metadata(self, documents: List[Document]):      
        if documents:
            print(f"Adding {len(documents)} documents with metadata...")
            
            await self.vs_repo.aadd_documents(documents)
            
            print(f"Successfully added {len(documents)} documents")
            
            sources = set(doc.metadata.get('source', 'unknown') for doc in documents)
            print(f"Sources: {list(sources)}")
        else:
            print("No documents to add")

    async def delete_documents_by_metadata(self, metadata: dict):
        search_results = await self.vs_repo.asimilarity_search(
            query="test",  
            k=1000,
            filter=metadata
        )
        log.info(f"deleted result {search_results}")
        ...

    #
    # RAG
    # 
    async def search(self, 
        query = "What is LangGraph?", 
        k = 10, 
        score_threshold: float = 0.7,
        metadata_filter = None
    ):
        if not query:
            return []

        try:
            if metadata_filter:
                results = await self.vs_repo.asimilarity_search_with_score(
                    query, 
                    k=k, 
                    filter=metadata_filter
                )
            else:
                results = await self.vs_repo.asimilarity_search_with_score(
                    query, 
                    k=k
                )
            
            formatted_results = []
            for doc, score in results:
                print(f"score thres: {score_threshold}, score: {score}")
                if score < score_threshold:
                    log.info("Check score\n")
                    if doc.metadata is None:
                        doc.metadata = {}
                    
                    doc.metadata['vector_distance'] = float(score)
                    
                    doc_id = doc.metadata.get('id', None)
                    
                    formatted_result = {
                        "id": doc_id,
                        "metadata": doc.metadata,
                        "page_content": doc.page_content,
                        "score": float(score)
                        # "type": "Document"
                    }
                    
                    formatted_results.append(formatted_result)
                
                log.info(f"Found {len(formatted_results)} results for query: '{query[:50]}...'\n")
            return formatted_results
            
        except Exception as e:
            print(f"Search error: {e}")
            return []
        ...

    async def manual_rag(
        self, 
        query: str,
        provider, 
        k: int = 3,
        score_threshold: float = 0.7,
        metadata_filter: Optional[Dict[str, Any]] = None,
        system_prompt: Optional[str] = None,
        include_metadata: bool = True
    ):
        system_prompt = """
        Bạn là trợ lý AI chuyên nghiệp.

        Hãy trả lời câu hỏi của người dùng dựa trên context
        """

        more = """
        YÊU CẦU BẮT BUỘC:
        1. Trả lời câu hỏi dựa trên ngữ cảnh
        2. KẾT THÚC câu trả lời với 3 dòng thông tin nguồn:

        Trong ngữ cảnh có metadata ở cuối mỗi tài liệu với định dạng:
        (Source: <tên file>, Chunk: <chunk id>, Type: <loại file>, Relevance Score: <điểm liên quan>)

        Hãy trích xuất thông tin từ metadata này và trình bày lại theo định dạng sau:

        - Nguồn: <tên file>
        - Trang: <không xác định nếu không có thông tin>

        QUAN TRỌNG:
        - Chỉ sử dụng thông tin từ metadata.
        - Nếu không có thông tin trang thì ghi: "không xác định".
        - Không sử dụng định dạng khác.

        Ví dụ output:

        - Nguồn: example.pdf
        - Trang: không xác định
        """
        system_prompt += more

        search_results = await self.search(
            query=query,
            k=k,
            score_threshold=score_threshold,
            metadata_filter=metadata_filter
        )

        context = self._format_context(
            search_results=search_results,
            include_metadata=include_metadata
        )
        log.info(f"Context: {context}")
        
        template = f"""{system_prompt}

        context: {context}

        câu hỏi: {query}

        hãy trả lời câu hỏi dựa trên context và metadata
        """
        log.info(f"Template: {template}")

        if search_results is None:
            return ""

        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | provider 
        answer = await chain.ainvoke({"query": query})

        sources = []
        for result in search_results:
            metadata = result["metadata"]
            sources.append({
                "source": metadata.get('source', 'unknown'),
                "chunk_index": metadata.get('source', 'unknown'),
                "page_number": metadata.get('page_number', 'unknown'),
                "type": metadata.get('type', 'unknown'),
                "relevance": result["score"]
            })
        
        return {
            "answer": answer.content,
            "query": query,
            "context": context,
            "sources": sources,
            "document_count": len(search_results),
            "score_threshold": score_threshold,
            "filter_applied": metadata_filter
        }
        ...

    def _format_context(
        self,
        search_results: List[Dict[str, Any]],
        include_metadata: bool = True,
        max_context_length: int = 4000
    ):
        if not search_results:
            return "No relevant documents found."

        context_parts = []
        current_length = 0

        for i, result in enumerate(search_results):
            doc_content = result["page_content"]
            score = result["score"]
            metadata = result["metadata"]
            
            doc_text = f"Document {i+1}:\n{doc_content}"
            
            if include_metadata and metadata:
                metadata_info = []
                
                if metadata.get('source'):
                    metadata_info.append(f"Source: {metadata['source']}")
                if metadata.get('chunk_index') is not None:
                    metadata_info.append(f"Chunk: {metadata['chunk_index']}")
                if metadata.get('page_number') is not None:
                    metadata_info.append(f"Page: {metadata['page_number']}")
                if metadata.get('type'):
                    metadata_info.append(f"Type: {metadata['type']}")
                metadata_info.append(f"Relevance Score: {score:.3f}")
                
                if metadata_info:
                    doc_text += f"\n({', '.join(metadata_info)})"
            
            if current_length + len(doc_text) > max_context_length:
                print(f"⚠️ Context truncated at {i} documents")
                break
            
            context_parts.append(doc_text)
            current_length += len(doc_text)
        
        formatted_context = "\n\n".join(context_parts)
        
        log.info(f"📝 Formatted context: {len(context_parts)} documents, {len(formatted_context)} chars")
        
        return formatted_context
    ...