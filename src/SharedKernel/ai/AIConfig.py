from abc import ABC, abstractmethod
import os
from typing import Any, Dict, Type
from SharedKernel.persistence.Decorators import Service
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_ollama import ChatOllama, OllamaEmbeddings
from src.SharedKernel.utils.yamlenv import load_env_yaml

config = load_env_yaml()

class AIConfig(ABC):
    @abstractmethod
    def create_provider(self) -> Any:
        pass

    @abstractmethod
    def create_embedding(self) -> Any:
        pass

class AIConfigFactory:
    _registry: Dict[str, Type[AIConfig]] = {}

    @classmethod
    def register(cls, type_name: str, ai_class: Type[AIConfig]):
        cls._registry[type_name] = ai_class

    @classmethod
    def create(cls, type_name: str) -> AIConfig:
        ai_config_class = cls._registry.get(type_name)
        if not ai_config_class:
            raise ValueError(f"AI Config '{type_name}' is not registered.")
        return ai_config_class()

class MistralConfig(AIConfig):
    def __init__(self) -> None:
        self.model = config.ai.mistral.model
        self.api_key = config.ai.mistral.api_key
        self.embeddings = config.ai.mistral.embed

    def create_provider(self):
        return ChatMistralAI(
            model=self.model,
            api_key=self.api_key
        )

    def create_embedding(self):
        embeddings = MistralAIEmbeddings(
            model=self.embeddings,
            api_key=self.api_key
        )
        return embeddings 

class OllamaConfig(AIConfig):
    def __init__(self) -> None:
        self.model = config.ai.ollama.model
        self.embeddings = config.ai.ollama.embed

    def create_provider(self):
        return ChatOllama(
            model=self.model,
            base_url=config.ai.ollama.host
        )

    def create_embedding(self):
        embeddings = OllamaEmbeddings(
            model=self.embeddings,
            base_url=config.ai.ollama.host
        )
        return embeddings 

AIConfigFactory.register("mistral", MistralConfig)
AIConfigFactory.register("ollama", OllamaConfig)