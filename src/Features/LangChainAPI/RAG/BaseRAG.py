from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator
from langchain_core.language_models.chat_models import BaseChatModel
from langchain.embeddings import Embeddings
from fastapi import UploadFile
from src.SharedKernel.base.Metrics import Metrics
from src.Features.LangChainAPI.RAG.Loader import Loader
from src.Features.LangChainAPI.RAG.Process import Process
from src.Features.LangChainAPI.persistence.MemoryRepository import MemoryRepository


class BaseRAG(ABC):
    def __init__(
        self,
        provider: BaseChatModel,
        embedding: Embeddings
    ) -> None:
        self.provider = provider
        self.embedding = embedding
        self.loader = Loader()
        self.process = Process()
        self.memory_repo = MemoryRepository()

    @abstractmethod
    async def index(self, file: UploadFile, **kwargs) -> None:
        """Ingest document vào retriever"""
        pass

    @abstractmethod
    async def retrieve(self, query: str, session_id: str = None, **kwargs) -> AsyncGenerator[str, None]:
        """Retrieve và generate response"""
        pass

    @abstractmethod
    async def delete(self, identifier: str, **kwargs) -> None:
        """Delete documents theo identifier (file_name/source)"""
        pass
