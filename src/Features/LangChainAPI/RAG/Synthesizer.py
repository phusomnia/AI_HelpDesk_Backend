import logging
import json
from typing import Any, Dict, List
from langchain_core.prompts import ChatPromptTemplate
from SharedKernel.ai.AIConfig import AIConfigFactory
from SharedKernel.utils.yamlenv import load_env_yaml
from fastapi import UploadFile
from src.Features.LangChainAPI.LangChainDTO import RagRequest
from src.Features.LangChainAPI.RAG.Loader import Loader
from src.Features.LangChainAPI.RAG.Process import Process
from src.Features.LangChainAPI.persistence.MemoryRepository import MemoryRepository
from src.Features.LangChainAPI.persistence.RedisVSRepository import RedisVSRepository

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
config = load_env_yaml()

class Synthesizer:
    def __init__(self, ai_factory: AIConfigFactory, provider) -> None:
        self.provider = provider
        self.ai_factory = ai_factory
        self.loader = Loader()
        self.process = Process()
        self.redis_vs_repo = RedisVSRepository(self.ai_factory)
        self.memory_repo = MemoryRepository()
    ...
    async def ingest_pdf_PaC(self, file: UploadFile):
        await self.redis_vs_repo.delete_documents_by_metadata(
            {"source": file.filename}
        )
        docs = self.loader.load_pdf(file)
        if not docs:
            print("No documents loaded")
            return
        chunks = self.process.split_PaC_v2(docs)
        await self.redis_vs_repo.add_PaC_documents_v2(chunks)
    
    async def delete_document_by_file_name(self, file_name: str):
        await self.redis_vs_repo.delete_documents_by_metadata(
            {"source": file_name}
        )
        
    async def rag_PaC(
        self,
        query: str,
        session_id: str = None
    ):
        if session_id:
            await self.memory_repo.add_message(
                session_id=session_id,
                role="user",
                content=query
            )
        
        hybrid_docs = await self.redis_vs_repo.hybrid_retriver_v2(query=query,k=10)
        print("Hybryd docs:", hybrid_docs)
        context = self._format_context_PaC_v2(hybrid_docs)

        # Build prompt
        system_prompt = """
        Bạn là trợ lý AI chuyên nghiệp.

        Hãy trả lời câu hỏi của người dùng dựa trên context

        YÊU CẦU BẮT BUỘC:
        1. Trả lời câu hỏi dựa trên ngữ cảnh
        2. KẾT THÚC câu trả lời với 3 dòng thông tin nguồn:

        Trong ngữ cảnh có metadata ở cuối mỗi tài liệu với định dạng:
        Source: <tên file>, Page: <trang>

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
        
        template = f"""{system_prompt}

        Ngữ cảnh: {context}

        Câu hỏi: {query}

        Hãy trả lời câu hỏi dựa trên ngữ cảnh

        Lưu ý nếu không tìm thấy thông tin thì output: tôi không có thông tin vui lòng liên hệ bộ phận hỗ trợ
        """
        # print(context)
        prompt = ChatPromptTemplate.from_template(template)

        chain = prompt | self.provider

        answer_parts = []
        async for chunk in chain.astream({"query": query}):
            if hasattr(chunk, "content"):
                answer_parts.append(chunk.content)

        async def response():
            answer_parts = []
            async for chunk in chain.astream({"query": query}):
                if hasattr(chunk, "content"):
                    answer_parts.append(chunk.content)
                    yield chunk.content
            
            answer = "".join(answer_parts)
            if session_id:
                await self.memory_repo.add_message(
                    session_id=session_id,
                    role="assistant",
                    content=answer
                )

        return response()

    # ============================================================
    # HELPER METHODS (Formatting & Extraction)
    # ============================================================

    def _format_context_PaC(self, search_results: List[Dict[str, Any]]) -> str:
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

    def _format_context_PaC_v2(self, search_results: List[Dict[str, Any]]) -> str:
        if not search_results:
            return "No relevant documents found."

        context_parts = []
        seen_parents = set()

        for idx, result in enumerate(search_results):
            parent_id = result.get("id")
            if not parent_id:
                continue

            if parent_id in seen_parents:
                continue
            seen_parents.add(parent_id)

            parent_content = result.get("content", "")
            parent_metadata = result.get("metadata", {})

            file_name = parent_metadata.get("source", "")
            pages = parent_metadata.get("pages", [])
            pages_str = ", ".join([str(p) for p in pages]) if pages else "không xác định"

            doc_content = parent_content.replace("\n", " ").strip()

            metadata_info = []
            if file_name:
                metadata_info.append(f"Source: {file_name}")
            metadata_info.append(f"Page: {pages_str}")

            doc_content = doc_content + "\n" + (" | ".join(metadata_info))
            context_parts.append(doc_content)

        formatted_context = "\n\n".join(context_parts)
        return formatted_context




