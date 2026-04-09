from abc import ABC, abstractmethod
from typing import Any, Dict, Type
from SharedKernel.base.Logger import get_logger
from SharedKernel.utils.yamlenv import load_env_yaml, load_redis_index
from langchain_core.embeddings.embeddings import Embeddings
from langchain_core.vectorstores import InMemoryVectorStore, VectorStore
from redisvl.schema import IndexSchema
from redis.client import Redis as RedisClient
from langchain_redis import RedisVectorStore
import uuid6

log = get_logger(__name__)
config: dict[str, Any] = load_env_yaml()
redis_config: dict[str, Any] = load_redis_index()


class VectoreStoreConfig(ABC):
    @abstractmethod
    def get_vecstore(self, embedding: Embeddings) -> VectorStore:
        pass

    @abstractmethod
    def get_url(self) -> str:
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


"""
Config Redis Vector Store
"""

class RedisVSManager(VectoreStoreConfig):
    def __init__(self) -> None:
        self.redis_url: str = config.redis.url
        self.index_name = redis_config["index"]["name"]
        self.prefix = redis_config["index"]["prefix"]

        log.info(
            f"Redis config - URL: {self.redis_url}, Index: {self.index_name}, Prefix: {self.prefix}"
        )

    def _check_index(self, client: RedisClient, index_name: str):
        client.execute_command("FT.INFO", index_name)
        log.info(f"Index: '{index_name}' exists")

    def get_url(self):
        return self.redis_url

    def get_vecstore(self, embedding: Embeddings) -> VectorStore:
        index_schema = IndexSchema.from_dict(redis_config)
        vector_store = None
        try:
            self._check_index(RedisClient.from_url(self.redis_url), self.index_name)
            vector_store = RedisVectorStore.from_existing_index(
                embedding=embedding,
                redis_url=self.redis_url,
                index_name=self.index_name,
            )
            log.info(f"✅ Redis vector store with index: {self.index_name}")
        except Exception as e:
            log.warning(f"{e}")

            RedisVectorStore(
                embeddings=embedding,
                redis_url=self.redis_url,
                index_schema=index_schema,
            )
            vector_store = RedisVectorStore.from_existing_index(
                embedding=embedding,
                redis_url=self.redis_url,
                index_name=self.index_name,
            )
            log.info(f"✅ Created new index: {self.index_name}")
        return vector_store


class InMemVSManager(VectoreStoreConfig):
    def __init__(self) -> None:
        pass

    def create_vector_store(self, embedding: Embeddings) -> VectorStore:
        return InMemoryVectorStore(embedding=embedding)
        ...


VectoreStoreConfigFactory.register("in_mem", InMemVSManager)
VectoreStoreConfigFactory.register("redis", RedisVSManager)
