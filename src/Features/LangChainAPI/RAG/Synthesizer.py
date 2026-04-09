import logging
import json
from typing import Any, Dict, List
from langchain_core.prompts import ChatPromptTemplate
from SharedKernel.config.LLMConfig import EmbeddingFactory
from src.SharedKernel.utils.yamlenv import load_env_yaml
from src.SharedKernel.base.Metrics import Metrics
from fastapi import UploadFile
from src.Features.LangChainAPI.RAG.Loader import Loader
from src.Features.LangChainAPI.RAG.Process import Process
from src.Features.LangChainAPI.RAG.LexicalGraphBuilder import LexicalGraphBuilder
from src.Features.LangChainAPI.persistence.MemoryRepository import MemoryRepository
from src.Features.LangChainAPI.persistence.RedisVSRepository import RedisVSRepository
from src.Features.LangChainAPI.persistence.Neo4JStore import Neo4JStore
from SharedKernel.threading.ThreadPoolManager import ThreadPoolManager
import asyncio

log = logging.getLogger(__name__)
config = load_env_yaml()


class Synthesizer:
    def __init__(
        self, 
        embedding_factory: type[EmbeddingFactory], 
        provider
    ) -> None:
        self.provider = provider
        self.embedding_factory = embedding_factory
        self.loader = Loader()
        self.process = Process()
        self._redis_vs_repo = None
        self._neo4j_store = None
        self._lexical_builder = None
        self.memory_repo = MemoryRepository()

        self.embedding_model = self.embedding_factory.create(config.llm.provider)

    @property
    def redis_vs_repo(self):
        if self._redis_vs_repo is None:
            self._redis_vs_repo = RedisVSRepository(self.embedding_factory)
        return self._redis_vs_repo

    @property
    def neo4j_store(self):
        if self._neo4j_store is None:
            self._neo4j_store = Neo4JStore(embedding_model=self.embedding_model)
        return self._neo4j_store

    @property
    def lexical_builder(self):
        if self._lexical_builder is None:
            self._lexical_builder = LexicalGraphBuilder(
                process=self.process,
                embedding_model=self.embedding_model,
                llm_provider=self.provider,
                neo4j_store=self.neo4j_store,
            )
        return self._lexical_builder

    ...

    async def ingest_file_PaC(self, file: UploadFile):
        metrics = Metrics("IngestPDF")

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

    async def build_graph(self, file: UploadFile, source: str):
        """Build Lexical Graph từ file - có metrics + logging"""
        metrics = Metrics("BuildLexicalGraph")
        log.info(f"======= Starting LexicalGraph: {source} =======")

        try:
            with metrics.stage("load_documents"):
                docs = await self.thread_pool.run_in_executor(
                    self.loader.load_file, file
                )
            metrics.increment("documents_count", len(docs))
            log.info(f"✓ Loaded {len(docs)} documents")

            with metrics.stage("build_graph"):
                result = await self.lexical_builder.build_graph(docs, source)

            metrics.increment("sections_count", result.get("sections", 0))
            metrics.increment("chunks_count", result.get("chunks", 0))
            metrics.increment("entities_count", result.get("entities", 0))
            metrics.increment("nodes_count", result.get("nodes", 0))
            metrics.increment("edges_count", result.get("edges", 0))

            log.info(f"✓ Graph built: {result}")

            with metrics.stage("store_neo4j"):
                log.info(f"✓ Stored to Neo4j")

            metrics.log_summary()
            log.info(f"======= Completed LexicalGraph: {source} =======")

            return result

        except Exception as e:
            log.error(f"LexicalGraph pipeline failed: {str(e)}")
            metrics.increment("error_count", 1)
            metrics.log_summary()
            raise

    async def query_graph_rag(self, question: str, source: str = None):
        """Graph RAG query pipeline - có metrics + logging"""
        metrics = Metrics("GraphRAGQuery")
        log.info(f"Starting Graph RAG query: {question}")

        try:
            with metrics.stage("extract_entities"):
                entities = await self.extract_query_entities(question)
            metrics.increment("entities_count", len(entities))
            log.info(f"Extracted {len(entities)} entities: {entities}")

            with metrics.stage("vector_search"):
                seed_chunks = await self.neo4j_store.search_by_embedding(
                    question, top_k=5
                )
            metrics.increment("seed_chunks_count", len(seed_chunks))

            with metrics.stage("subgraph_traversal"):
                facts = await self.traverse_subgraph(
                    [c["node_id"] for c in seed_chunks], depth=2
                )
            metrics.increment("facts_count", len(facts))

            with metrics.stage("section_summaries"):
                section_summaries = await self.get_section_summaries(
                    [c["node_id"] for c in seed_chunks]
                )

            with metrics.stage("document_summary"):
                document_summary = (
                    await self.get_document_summary(source) if source else ""
                )

            with metrics.stage("llm_generation"):
                answer = await self.generate_answer(
                    question=question,
                    seed_chunks=seed_chunks,
                    facts=facts,
                    section_summaries=section_summaries,
                    document_summary=document_summary,
                )

            metrics.log_summary()
            log.info(f"Graph RAG query completed")

            return {"answer": answer, "sources": seed_chunks, "entities": entities}

        except Exception as e:
            log.error(f"Graph RAG query failed: {str(e)}")
            metrics.increment("error_count", 1)
            metrics.log_summary()
            raise

    async def extract_query_entities(self, question: str) -> List[str]:
        """Extract entities từ câu hỏi bằng LLM"""
        prompt = f"""
            Extract entities from the question: {question}

            Return list of entity names only (one per line):
        """
        try:
            response = self.provider.invoke(prompt)
            entities = [e.strip() for e in response.content.split("\n") if e.strip()]
            return entities
        except Exception as e:
            log.error(f"Entity extraction failed: {e}")
            return []

    async def traverse_subgraph(
        self, chunk_ids: List[str], depth: int = 2
    ) -> List[Dict]:
        """Traverse relations để lấy related entities và facts"""
        facts = []

        for chunk_id in chunk_ids:
            neighbors = await self.neo4j_store.get_neighbors(chunk_id, depth=depth)
            facts.extend(neighbors)

        return facts

    async def get_section_summaries(self, chunk_ids: List[str]) -> List[str]:
        """Lấy section summaries cho hierarchical context"""
        summaries = []

        for chunk_id in chunk_ids:
            section = await self.neo4j_store.get_parent_section(chunk_id)
            if section:
                summaries.append(section.get("summary", ""))

        return summaries

    async def get_document_summary(self, source: str) -> str:
        """Lấy global document summary"""
        return await self.neo4j_store.get_document_summary(source)

    async def generate_answer(
        self,
        question: str,
        seed_chunks: List[Dict],
        facts: List[Dict],
        section_summaries: List[str],
        document_summary: str,
    ) -> str:
        """Generate answer từ context"""

        context = f"""
            Document Summary:
            {document_summary}

            Section Summaries:
            {chr(10).join(section_summaries)}

            Relevant Facts:
            {chr(10).join([str(f) for f in facts])}

            Seed Chunks:
            {chr(10).join([c.get("content", "") for c in seed_chunks])}
            """

        prompt = f"""
            Based on the following context, answer the question.

            Context:
            {context}

            Question: {question}

            Answer:
            """

        try:
            response = self.provider.invoke(prompt)
            return response.content
        except Exception as e:
            log.error(f"LLM generation failed: {e}")
            return "Xin lỗi, tôi không thể trả lời câu hỏi này."

    async def delete_document_by_file_name(self, file_name: str):
        await self.redis_vs_repo.delete_documents_by_metadata({"source": file_name})

    async def retriver_documents_PaC(self, query: str, session_id: str = None):
        metrics = Metrics("Retriever")

        if session_id:
            with metrics.stage("memory_add_user"):
                await self.memory_repo.add_message(
                    session_id=session_id, role="user", content=query
                )

        with metrics.stage("hybrid_retrieval"):
            hybrid_docs = await self.redis_vs_repo.hybrid_retriver(query=query, k=5)
        metrics.increment("retrieved_docs", len(hybrid_docs))

        with metrics.stage("context_formatting"):
            context = self._format_context_PaC(hybrid_docs)
            log.info(context)
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
        metrics.increment("answer_tokens", len("".join(answer_parts).split()))

        metrics.log_summary()

        async def response():
            answer_parts = []
            async for chunk in chain.astream({"query": query}):
                if hasattr(chunk, "content"):
                    answer_parts.append(chunk.content)
                    yield chunk.content

            answer = "".join(answer_parts)

            if session_id:
                with metrics.stage("memory_add_assistant"):
                    await self.memory_repo.add_message(
                        session_id=session_id, role="assistant", content=answer
                    )
                metrics.log_summary()

        return response()

    # ============================================================
    # HELPER METHODS (Formatting & Extraction)
    # ============================================================
    def _format_context_PaC(self, search_results: List[Dict[str, Any]]) -> str:
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
