import logging
from typing import Any, Dict, List, AsyncGenerator
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.embeddings import Embeddings
from fastapi import UploadFile
from SharedKernel.base.Metrics import Metrics
from Features.LangChainAPI.RAG.BaseRAG import BaseRAG
from Features.LangChainAPI.persistence.RedisVSRepository import RedisVSRepository
from SharedKernel.config.LLMConfig import EmbeddingFactory

log = logging.getLogger(__name__)

class PaCRAG(BaseRAG):
    def __init__(
        self,
        provider: BaseChatModel,
        embedding: Embeddings
    ) -> None:
        super().__init__(provider, embedding)
        self._redis_vs_repo = None

    @property
    def redis_vs_repo(self):
        if self._redis_vs_repo is None:
            self._redis_vs_repo = RedisVSRepository(EmbeddingFactory)
        return self._redis_vs_repo

    async def index(self, file: UploadFile, **kwargs) -> None:
        """Ingest PDF file vào Redis vector store với Page-aware Chunking"""
        metrics = Metrics("Index PDF")

        with metrics.stage("delete_existing"):
            await self.redis_vs_repo.delete_documents_by_metadata(
                {"source": file.filename}
            )

        with metrics.stage("load_pdf"):
            docs = await self.loader.load_pdf(file)

        if not docs:
            print("No documents loaded")
            return

        with metrics.stage("split_pac"):
            chunks = await self.process.split_PaC(docs)

        with metrics.stage("add_documents"):
            await self.redis_vs_repo.add_PaC_documents(chunks)

        metrics.log_summary()

    async def retrieve(
        self,
        query: str,
        session_id: str = None,
        **kwargs
    ) -> AsyncGenerator[str, None]:
        """Retrieve documents và generate response với streaming"""
        metrics = Metrics("Retriever")

        if session_id:
            with metrics.stage("memory_add_user"):
                await self.memory_repo.add_message(
                    session_id=session_id, role="user", content=query
                )

        with metrics.stage("hybrid_retrieval"):
            hybrid_docs = await self.redis_vs_repo.hybrid_retriver(query=query, k=5)
            print(hybrid_docs)
        metrics.increment("retrieved_docs", len(hybrid_docs))

        with metrics.stage("context_formatting"):
            context = self._format_context_PaC(hybrid_docs)
            print(context)
        metrics.increment("context_length", len(context))

        with metrics.stage("prompt_building"):
            system_prompt = """
            Bạn là trợ lý AI chuyên nghiệp

            ## Quy tắc xử lý

            1. Trường hợp người dùng chỉ chào
            Nếu người dùng chỉ gửi lời chào (ví dụ: "xin chào", "hello", "hi", ...):

            - Chỉ chào lại một cách lịch sự.
            - **KHÔNG trả lời nội dung.**
            - **KHÔNG sử dụng ngữ cảnh.**
            - **KHÔNG hiển thị nguồn.**

            2. Trong trường hợp người dùng gửi câu hỏi thì:
            Hãy trả lời câu hỏi của người dùng dựa trên context

            YÊU CẦU BẮT BUỘC:
            1. Tuân theo quy tắc xử lý
            2. Trả lời câu hỏi dựa trên ngữ cảnh
            3. KẾT THÚC câu trả lời với 3 dòng thông tin nguồn:

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
            prompt = ChatPromptTemplate.from_template(template)
            chain = prompt | self.provider

        with metrics.stage("llm_generation"):
            answer_parts = []
            async for chunk in chain.astream({"query": query}):
                if hasattr(chunk, "content"):
                    answer_parts.append(chunk.content)
                    yield chunk.content
        metrics.increment("answer_tokens", len("".join(answer_parts).split()))

        answer = "".join(answer_parts)

        if session_id:
            with metrics.stage("memory_add_assistant"):
                await self.memory_repo.add_message(
                    session_id=session_id, role="assistant", content=answer
                )

        metrics.log_summary()

    async def delete(self, identifier: str, **kwargs) -> None:
        """Delete documents theo file_name"""
        await self.redis_vs_repo.delete_documents_by_metadata({"source": identifier})

    async def get_chat_history(self, session_id: str, **kwargs) -> List[Dict]:
        """Get chat history for a session"""
        return await self.memory_repo.get_history_all(session_id)

    def _format_context_PaC(self, search_results: List[Dict[str, Any]]) -> str:
        """Format search results thành context string cho PaC"""
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
            pages_str = (
                ", ".join([str(p) for p in pages]) if pages else "không xác định"
            )

            doc_content = parent_content.replace("\n", " ").strip()

            metadata_info = []
            if file_name:
                metadata_info.append(f"Source: {file_name}")
            metadata_info.append(f"Page: {pages_str}")

            doc_content = doc_content + "\n" + (" | ".join(metadata_info))
            context_parts.append(doc_content)

        formatted_context = "\n\n".join(context_parts)
        formatted_context = formatted_context.replace("{", "{{").replace("}", "}}")
        return formatted_context
