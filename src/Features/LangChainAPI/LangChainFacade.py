from fastapi import Depends
from SharedKernel.config.LLMConfig import LLMFactory, EmbeddingFactory
from SharedKernel.utils.yamlenv import load_env_yaml
from src.Features.LangChainAPI.RAG.Synthesizer import Synthesizer
from src.Features.LangChainAPI.service.AgentService import AgentService
from src.Features.LangChainAPI.service.PromptService import PromptService

class LangChainFacade:
    def __init__(self):       
        self.config = load_env_yaml()
        self.provider = LLMFactory.create(self.config.llm.provider)

        self.prompt_service = PromptService()
        self.agent_service = AgentService()
        self.synthesizer = Synthesizer(
            EmbeddingFactory, 
            self.provider 
        )
        ...
    

