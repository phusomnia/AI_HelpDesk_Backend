from typing import Any
from Features.LangChainAPI.LangChainDTO import Callback
from Features.LangChainAPI.service.Synthesis import Synthesis
from Features.LangChainAPI.service.MemoryService import MemoryService
from Features.LangChainAPI.service.OutputService import OutputService
from Features.LangChainAPI.service.PromptService import PromptService
from Features.LangChainAPI.service.ToolService import ToolService
from SharedKernel.ai.AIConfig import AIConfigFactory
from SharedKernel.persistence.Decorators import Service
from SharedKernel.utils.yamlenv import load_env_yaml

config = load_env_yaml()

# @Service
class LangChainFacade:
    def __init__(self,
        config_ai: AIConfigFactory
    ):       
        self.ai_factory = config_ai
        ai_config = self.ai_factory.create(config.ai.llm_provider)
        self.provider = ai_config.create_provider()

        self.callbacks = Callback(
            ainvoke=self.ainvoke,
            astream=self.astream
        )
        
        self.prompt = PromptService(self.provider, self.callbacks)
        self.tools = ToolService(self.provider, self.callbacks)
        self.output_parser = OutputService(self.provider, self.callbacks)
        self.memory = MemoryService(self.provider, self.callbacks)
        self.SYN = Synthesis(self.ai_factory, self.provider, self.callbacks)
    
    async def ainvoke(self, provider: Any, req: Any):
        result = await provider.ainvoke(req)
        if hasattr(result, 'content') and result.content:
            return result.content
        return result

    async def astream(self, provider: Any, req: Any) -> dict:
        async def gen(req):
            async for chunk in provider.astream(req):
                if chunk.content:
                    yield chunk.content

        return { "content": gen(req) }
    

