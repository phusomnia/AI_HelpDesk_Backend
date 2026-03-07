# from langchain_redis import RedisVectorStore
# from redis.asyncio import Redis
# import os

# from redisvl.schema import IndexSchema

# class VectorDBConfig:
#     redis_url: str = os.getenv("REDIS_URL")
#     vector_dim: int = os.getenv("VECTOR_DIM")

#     index_name: str = "vector_index"
#     prefix: str = "doc:"
#     vector_field: str = "embedding"
#     content_field: str = "content"
#     distance_metric: str = "COSINE"
#     index_method: str = "HNSW"

#     def __init__(self):
#         self._client = None
#         self._index_created = False

#     def client(self) -> Redis:
#         if self._client is None:
#             self._client = Redis.from_url(self.redis_url)
#             if not self._client.ping():
#                 raise ConnectionError("Could not connect to Redis")
#         return self._client

#     def index_schema(self):
#         return IndexSchema.from_dict({
#             "index": {
#                 "name": self.index_name,
#                 "prefix": self.prefix,
#             },
#             "fields": [
#                 {"name": self.content_field, "type": "text"},
#                 {
#                     "name": self.vector_field,
#                     "type": "vector",
#                     "attrs": {
#                         "dims": self.vector_dim,
#                         "distance_metric": self.distance_metric,
#                         "algorithm": self.index_method,
#                         "datatype": "float32"
#                     },
#                 },
#             ],
#         })

#     def create_vector_store(self, embeddings) -> RedisVectorStore:
#         return 1(
#             embeddings=embeddings,
#             redis_url=self.redis_url,
#             index_name=self.index_name,
#             index_schema=self.index_schema(),
#         )

