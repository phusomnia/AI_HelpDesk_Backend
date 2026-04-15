import logging
from typing import Any, Dict, List
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.embeddings import Embeddings
from fastapi import UploadFile
from src.SharedKernel.base.Metrics import Metrics
from src.Features.LangChainAPI.RAG.BaseRAG import BaseRAG
from src.Features.LangChainAPI.RAG.LexicalGraphBuilder import LexicalGraphBuilder
from src.Features.LangChainAPI.persistence.Neo4JStore import Neo4JStore
from SharedKernel.threading.ThreadPoolManager import ThreadPoolManager

log = logging.getLogger(__name__)


class GraphRAG(BaseRAG):
    def __init__(
        self,
        provider: BaseChatModel,
        embedding: Embeddings
    ) -> None:
        super().__init__(provider, embedding)
        self._neo4j_store = None
        self._lexical_builder = None
        self.thread_pool = ThreadPoolManager()

    @property
    def neo4j_store(self):
        if self._neo4j_store is None:
            self._neo4j_store = Neo4JStore(embedding_model=self.embedding)
        return self._neo4j_store

    @property
    def lexical_builder(self):
        if self._lexical_builder is None:
            self._lexical_builder = LexicalGraphBuilder(
                process=self.process,
                embedding_model=self.embedding,
                llm_provider=self.provider,
                neo4j_store=self.neo4j_store,
            )
        return self._lexical_builder

    async def ingest(self, file: UploadFile, source: str = None, **kwargs) -> None:
        """Build Lexical Graph từ file - có metrics + logging"""
        if not source:
            source = file.filename

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

    async def retrieve(
        self,
        query: str,
        session_id: str = None,
        source: str = None,
        **kwargs
    ) -> dict:
        """Graph RAG query pipeline - có metrics + logging"""
        metrics = Metrics("GraphRAGQuery")
        log.info(f"Starting Graph RAG query: {query}")

        try:
            with metrics.stage("extract_entities"):
                entities = await self.extract_query_entities(query)
            metrics.increment("entities_count", len(entities))
            log.info(f"Extracted {len(entities)} entities: {entities}")

            with metrics.stage("vector_search"):
                seed_chunks = await self.neo4j_store.search_by_embedding(
                    query, top_k=5
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
                    question=query,
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

    async def delete(self, identifier: str, **kwargs) -> None:
        """Delete nodes by source"""
        await self.neo4j_store.delete_by_source(identifier)

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
