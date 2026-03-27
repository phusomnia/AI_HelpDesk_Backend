import logging
from Features.LangChainAPI.repo.RedisVS import RedisVS
from SharedKernel.ai.AIConfig import AIConfigFactory
from SharedKernel.utils.yamlenv import load_env_yaml
from fastapi import UploadFile
from src.Features.LangChainAPI.LangChainDTO import RagRequest
from src.Features.LangChainAPI.RAG.Loader import Loader
from src.Features.LangChainAPI.RAG.Process import Process

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)
config = load_env_yaml()

# @Service
class Synthesis:
    def __init__(self, ai_factory: AIConfigFactory, provider, callbacks) -> None:
        self.provider = provider
        self.callbacks = callbacks or {}
        self.ai_factory = ai_factory
        self.loader = Loader(self.provider, self.callbacks)
        self.process = Process(self.provider, self.callbacks)
        self.vs_repo = RedisVS(self.ai_factory)
    ...

    async def call_rag(self, req: RagRequest): 
        result = await self.vs_repo.rag_PaC(req.query, self.provider)
        return result
        ...

    #
    # INGEST
    #
    """

    """
    async def ingest_pdf_PaC(self, file: UploadFile):
        await self.vs_repo.delete_documents_by_metadata(
            {"source": file.filename}
        )
        docs = self.loader.load_pdf(file)
        if not docs:
            print("No documents loaded")
            return
        chunks = self.process.split_PaC(docs)
        await self.vs_repo.add_PaC_documents(chunks)
    ...

###


