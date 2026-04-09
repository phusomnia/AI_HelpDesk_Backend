import logging
from typing import Any, Dict, List, Optional
from SharedKernel.persistence.Neo4jManager import Neo4jManager, get_neo4j_manager

log = logging.getLogger(__name__)


class Neo4JStore(Neo4jManager):
    """Store lexical graph vào Neo4j - dùng Neo4jManager singleton"""
    # def __new__(cls, embedding_model=None):
    #     return super().__new__(cls)

    def __init__(self, embedding_model=None):
        super().__init__()
        self.embedding_model = embedding_model
        self._index_initialized = False

    def init_vector_index(self, index_name: str = "chunk_embeddings"):
        """Tạo vector index nếu chưa có"""
        if self._index_initialized:
            return

        self.execute_query(f"""
            CREATE INDEX {index_name} IF NOT EXISTS
            FOR (n:Chunk) ON (n.embedding)
        """)
        self._index_initialized = True
        log.info(f"Initialized vector index: {index_name}")

    async def add_graph(self, nodes: List[Dict], edges: List[Dict]):
        """Add nodes và edges vào Neo4j"""
        log.info(f"Adding {len(nodes)} nodes and {len(edges)} edges to Neo4j")

        for node in nodes:
            self._add_node(node)

        for edge in edges:
            self._add_edge(edge)

        log.info("Graph added successfully")

    def _add_node(self, node: Dict):
        """Add single node"""
        node_type = node.get("type", "Node")
        node_id = node.get("id", "")

        if node_type == "Chunk":
            self.execute_query(
                """
                MERGE (c:Chunk {id: $id})
                SET c.content = $content, c.embedding = $embedding
            """,
                {
                    "id": node_id,
                    "content": node.get("content", ""),
                    "embedding": node.get("embedding", []),
                },
            )

        elif node_type == "Section":
            self.execute_query(
                """
                MERGE (s:Section {id: $id})
                SET s.content = $content, s.summary = $summary
            """,
                {
                    "id": node_id,
                    "content": node.get("content", ""),
                    "summary": node.get("summary", ""),
                },
            )

        elif node_type == "Entity":
            self.execute_query(
                """
                MERGE (e:Entity {id: $id})
                SET e.name = $name, e.entity_type = $entity_type
            """,
                {
                    "id": node_id,
                    "name": node.get("name", ""),
                    "entity_type": node.get("entity_type", ""),
                },
            )

    def _add_edge(self, edge: Dict):
        """Add single edge"""
        source = edge.get("source", "")
        target = edge.get("target", "")
        edge_type = edge.get("type", "RELATED")

        if edge_type == "CONTAINS":
            self.execute_query(
                """
                MATCH (s:Section {id: $source})
                MATCH (c:Chunk {id: $target})
                MERGE (s)-[:CONTAINS]->(c)
            """,
                {"source": source, "target": target},
            )

        elif edge_type == "SUMMARIZES":
            self.execute_query(
                """
                MATCH (s:Section {id: $source})
                MATCH (e:Entity {id: $target})
                MERGE (s)-[:SUMMARIZES]->(e)
            """,
                {"source": source, "target": target},
            )

        elif edge_type == "EMBEDS":
            self.execute_query(
                """
                MATCH (c:Chunk {id: $source})
                MATCH (e:Entity {id: $target})
                MERGE (c)-[:EMBEDS]->(e)
            """,
                {"source": source, "target": target},
            )

        elif edge_type == "REFERENCES":
            self.execute_query(
                """
                MATCH (e1:Entity {id: $source})
                MATCH (e2:Entity {id: $target})
                MERGE (e1)-[:REFERENCES]->(e2)
            """,
                {"source": source, "target": target},
            )

    async def search_by_embedding(self, query: str, top_k: int = 5) -> List[Dict]:
        """Tìm chunks tương tự bằng cosine similarity"""
        if not self.embedding_model:
            log.error("Embedding model not set")
            return []

        query_embedding = self.embedding_model.embed_query(query)

        results = self.execute_query(
            """
            MATCH (c:Chunk)
            WHERE c.embedding IS NOT NULL
            WITH c, reduce(dot = 0.0, i IN range(0, size(c.embedding)-1) | 
                dot + c.embedding[i] * $query_emb[i]) AS dot_product,
                 reduce(sq = 0.0, i IN range(0, size(c.embedding)-1) | 
                sq + c.embedding[i] * c.embedding[i]) AS c_norm_sq,
                 reduce(sq = 0.0, i IN range(0, size($query_emb)-1) | 
                sq + $query_emb[i] * $query_emb[i]) AS q_norm_sq
            WITH c, dot_product / (sqrt(c_norm_sq) * sqrt(q_norm_sq)) AS score
            WHERE score > 0
            RETURN c.id AS node_id, c.content AS content, score
            ORDER BY score DESC
            LIMIT $top_k
        """,
            {"query_emb": query_embedding, "top_k": top_k},
        )

        return results

    async def get_neighbors(self, node_id: str, depth: int = 2) -> List[Dict]:
        """Lấy các nodes liên quan"""
        results = self.execute_query(
            """
            MATCH path = (start)-[*1..{depth}]-(end)
            WHERE start.id = $node_id
            UNWIND NODES(path) AS node
            RETURN DISTINCT node.id AS node_id, node.content AS content,
                   node.entity_type AS entity_type
        """,
            {"node_id": node_id, "depth": depth},
        )

        return results

    async def get_parent_section(self, chunk_id: str) -> Optional[Dict]:
        """Lấy parent section của chunk"""
        results = self.execute_query(
            """
            MATCH (s:Section)-[:CONTAINS]->(c:Chunk {id: $chunk_id})
            RETURN s.id AS section_id, s.summary AS summary, s.content AS content
            LIMIT 1
        """,
            {"chunk_id": chunk_id},
        )

        return results[0] if results else None

    async def get_document_summary(self, source: str) -> str:
        """Lấy document summary"""
        results = self.execute_query(
            """
            MATCH (s:Section)
            WHERE s.content CONTAINS $source
            RETURN COLLECT(s.summary) AS summaries
        """,
            {"source": source},
        )

        if results and results[0].get("summaries"):
            return " | ".join(results[0]["summaries"])
        return ""

    async def get_graph_stats(self, source: str = None) -> Dict:
        """Lấy thống kê graph"""
        if source:
            nodes = self.execute_query(
                """
                MATCH (n)
                WHERE n.content CONTAINS $source OR n.metadata.source = $source
                RETURN COUNT(n) AS node_count
            """,
                {"source": source},
            )
        else:
            nodes = self.execute_query("""
                MATCH (n) RETURN COUNT(n) AS node_count
            """)

        chunks = self.execute_query("""
            MATCH (c:Chunk) RETURN COUNT(c) AS count
        """)
        sections = self.execute_query("""
            MATCH (s:Section) RETURN COUNT(s) AS count
        """)
        entities = self.execute_query("""
            MATCH (e:Entity) RETURN COUNT(e) AS count
        """)

        return {
            "node_count": nodes[0].get("node_count", 0) if nodes else 0,
            "chunks": chunks[0].get("count", 0) if chunks else 0,
            "sections": sections[0].get("count", 0) if sections else 0,
            "entities": entities[0].get("count", 0) if entities else 0,
        }

    async def delete_graph(self, source: str = None):
        """Xóa graph"""
        if source:
            self.execute_query(
                """
                MATCH (n)
                WHERE n.content CONTAINS $source
                DETACH DELETE n
            """,
                {"source": source},
            )
        else:
            self.execute_query("""
                MATCH (n) DETACH DELETE n
            """)
        log.info(f"Deleted graph for source: {source}")


def get_neo4j_store(embedding_model=None) -> Neo4JStore:
    return Neo4JStore(embedding_model=embedding_model)
