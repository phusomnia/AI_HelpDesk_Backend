from abc import ABC, abstractmethod
import logging
from typing import Dict, Type
from SharedKernel.utils.yamlenv import load_env_yaml
from langchain_core.embeddings.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore
from redisvl.schema import IndexSchema
from redis.client import Redis as RedisClient
from langchain_redis import RedisVectorStore
import uuid6

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
config = load_env_yaml()

class VectoreStoreConfig(ABC):
    @abstractmethod
    def get_vecstore(self, embedding: Embeddings) -> VectorStore:
        pass

class VectoreStoreConfigFactory:
    _registry: Dict[str, Type[VectoreStoreConfig]] = {}

    @classmethod
    def register(cls, type_name: str, ai_class: Type[VectoreStoreConfig]):
        cls._registry[type_name] = ai_class

    @classmethod
    def create(cls, type_name: str) -> VectoreStoreConfig:
        ai_config_class = cls._registry.get(type_name)
        if not ai_config_class:
            raise ValueError(f"Vector store config '{type_name}' is not registered.")
        return ai_config_class()


new_id = uuid6.uuid7()

"""
Config Redis Vector Store
"""
class RedisVSManager(VectoreStoreConfig):
    def __init__(self) -> None:
        self.redis_url = config.redis.url
        self.index_name = getattr(config.vector_store.redis, 'index_name', 'default_idx')
        self.prefix = getattr(config.vector_store.redis, 'prefix', self.index_name)
        self.content_field = getattr(config.vector_store.redis, 'content_field', 'text')
        self.vector_field = getattr(config.vector_store.redis, 'vector_field', 'vector')
        self.distance_metric = getattr(config.vector_store.redis, 'distance_metric', 'COSINE')
        self.vector_dim = getattr(config.vector_store.redis, 'vector_dim', 1536)
        self.algorithm = getattr(config.vector_store.redis, 'algorithm', 'FLAT')

        log.info(f"Redis config - URL: {self.redis_url}, Index: {self.index_name}, Prefix: {self.prefix}")

    def _index_exists(self, client: RedisClient, index_name: str) -> bool:
        try:
            client.execute_command('FT.INFO', index_name)
            log.info(f"Index: '{index_name}' exists")
            return True
        except Exception as e:
            log.warning(f"Redis error checking index: {e}")
            return False

    def get_vecstore(self, embedding: Embeddings) -> VectorStore:
        """
        """

        client = RedisClient.from_url(self.redis_url)
        vector_store = None

        # if not self._index_exists(client, self.index_name):
        metadata_fields = getattr(config.vector_store.redis, 'metadata_fields', [])
        log.info(metadata_fields)

        index_schema = IndexSchema.from_dict({
            "index": {
                "name": self.index_name,  # Đảm bảo dùng tên này
                "prefix": self.prefix,
            },
            "fields": [
                {"name": self.content_field, "type": "text"}, 
                *metadata_fields,
                {
                    "name": self.vector_field,  
                    "type": "vector",
                    "attrs": {
                        "dims": self.vector_dim,  
                        "distance_metric": self.distance_metric,
                        "algorithm": self.algorithm,
                        "datatype": "FLOAT32"
                    },
                },
            ],
        })
        
        RedisVectorStore(
            embeddings=embedding,
            redis_url=self.redis_url,
            index_schema=index_schema,
        )
        
        print(f"✅ Redis vector store initialized: {self.index_name}")

        vector_store = RedisVectorStore.from_existing_index(
            embedding=embedding,
            redis_url=self.redis_url,
            index_name=self.index_name,
        )
        print(f"✅ Redis vector store with index: {self.index_name}")
        return vector_store

class InMemVSManager(VectoreStoreConfig):
    def __init__(self) -> None:
        pass

    def create_vector_store(self, embedding: Embeddings) -> VectorStore: 
        return InMemoryVectorStore(embedding=embedding)
        ...

VectoreStoreConfigFactory.register("in_mem", InMemVSManager)
VectoreStoreConfigFactory.register("redis", RedisVSManager)
