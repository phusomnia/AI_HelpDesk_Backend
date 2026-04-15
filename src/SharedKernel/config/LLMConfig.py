from abc import ABC, abstractmethod
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.embeddings import Embeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from typing import Dict, Type
from src.SharedKernel.utils.yamlenv import load_env_yaml

config = load_env_yaml()

#
# INTERFACE 
# 
class BaseLLMProvider(ABC):
    @abstractmethod
    def get_llm(self) -> BaseChatModel:
        pass

class BaseEmbeddingProvider(ABC):
    @abstractmethod
    def get_embedding(self) -> Embeddings:
        pass

#
# PROVIDER
#
class MistralProvider(BaseLLMProvider, BaseEmbeddingProvider):
    def __init__(self) -> None:
        self.model = config.llm.mistral.model
        self.api_key = config.llm.mistral.api_key
        self.embedding_model = config.llm.mistral.embed

    def get_llm(self) -> BaseChatModel:
        return ChatMistralAI(
            model=self.model,
            api_key=self.api_key,
            timeout=30,
            max_retries=3
        )

    def get_embedding(self):
        return MistralAIEmbeddings(
            model=self.embedding_model,
            api_key=self.api_key
        )

class OllamaProvider(BaseLLMProvider, BaseEmbeddingProvider):
    def __init__(self) -> None:
        self.model = config.llm.ollama.model
        self.host = config.llm.ollama.host
        self.embedding_model = config.llm.ollama.embed

    def get_llm(self) -> BaseChatModel:
        return ChatOllama(
            model=self.model,
            base_url=self.host
        )

    def get_embedding(self) -> Embeddings:
        return OllamaEmbeddings(
            model=self.embedding_model,
            base_url=self.host
        )

#
# REGISTRY
#
class ProviderRegistry:
    _providers: Dict[str, Type] = {}

    @classmethod
    def register(cls, name: str, provider: Type):
        cls._providers[name] = provider

    @classmethod
    def get(cls, name: str):
        provider = cls._providers.get(name)
        if not provider:
            raise ValueError(f"Provider '{name}' not found")
        return provider()

#
# FACTORY
# 
class LLMFactory:
    @staticmethod
    def create(provider_name: str) -> BaseChatModel:
        provider = ProviderRegistry.get(provider_name)
        return provider.get_llm()

class EmbeddingFactory:
    @staticmethod
    def create(provider_name: str) -> Embeddings:
        provider = ProviderRegistry.get(provider_name)
        return provider.get_embedding()

ProviderRegistry.register("mistral", MistralProvider)
ProviderRegistry.register("ollama", OllamaProvider)