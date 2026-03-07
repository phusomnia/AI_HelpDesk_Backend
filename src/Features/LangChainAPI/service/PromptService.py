from Features.LangChainAPI.LangChainDTO import Callback, ChatRequest, ChatTechniqueRequest, ChatTemplateRequest, TechType, TemplateType
from Features.LangChainAPI.prompt import COT_PROMPT, FEW_SHOT_PROMPT, REACT_PROMPT, ZERO_SHOT_PROMPT, System_Instruction
from SharedKernel.persistence.Decorators import Service
from SharedKernel.utils.yamlenv import load_env_yaml
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import config
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.output_parsers import StrOutputParser

@Service
class PromptService:
    def __init__(self, provider: BaseChatModel, callbacks: Callback):
        self.provider = provider
        self.callbacks = callbacks

    async def aprompt(self, req: ChatRequest):
        return await self.callbacks.ainvoke(self.provider, [System_Instruction(req)])
        pass

    async def asprompt(self, req: ChatRequest) -> dict:
        return await self.callbacks.astream(self.provider, [System_Instruction(req)])

    async def prompt_template(self, req: ChatTemplateRequest) -> dict:
        async def from_template():
            template = "Viết một blog ngắn nói về {topic}" 
            _prompt = PromptTemplate.from_template(template)
            prompt = _prompt.format(topic=req.message)

            return await self.callbacks.astream(self.provider, prompt)

        async def prompt_template():
            template = "Viết một blog ngắn nói về {topic} khoảng {words}"
            _prompt = PromptTemplate(
                # nội dung chính
                template=template,
                # biến đầu vào
                input_variables=['topic'],
                # định nghĩ kiểu dữ liệu
                input_types={},
                # biến cố định
                partial_variables={'words': '300 từ'}
            )
            prompt = _prompt.format(topic=req.message)

            # hoặc
            # prompt = _prompt.partial(words="300 từ")
            # prompt = _prompt.format(topic=req.message)
            # async for chunk in self.provider.astream(prompt):
            #     yield chunk

            return await self.callbacks.astream(self.provider, prompt)

        async def message_placeholder():
            template = ChatPromptTemplate.from_template("Viết một blog ngắn nói về {topic} khoảng {words}")
            chain = template | self.provider 
            placeholder = {
                "topic": req.message,
                "words": "300 từ"
            }

            return await self.callbacks.astream(chain, placeholder)

        async def chat_template():
            template = ChatPromptTemplate.from_messages([
                ("system", "Bạn là trợ lý AI"),
                MessagesPlaceholder(variable_name="hoi_thoai"),
                ("human", "Tóm tắt nội dung trong {so_tu} từ"),
            ])

            chain = template | self.model | StrOutputParser()

            placeholder = {
                "so_tu": 20,
                "hoi_thoai": [
                    ("human", 
                    """Xin chào bạn Teddy, tên bạn là gì thế mình quên rồi :D, 
                    Viết một blog ngắn nói về manga khoảng 200 từ"""
                    )
                ],
            }

            return await self.callbacks.astream(chain, placeholder)

        template_dict = {
            TemplateType.from_template: from_template(),
            TemplateType.prompt_template: prompt_template(), 
            TemplateType.chat_template: chat_template(),
            TemplateType.message_placeholder: message_placeholder()
        }

        return template_dict

    async def promt_techniques(self, req: ChatTechniqueRequest) -> dict:
        async def gen(message: str) -> dict:
            return await self.callbacks.astream(self.provider, message)

        techies_rsg = {
            TechType.ZERO: gen(ZERO_SHOT_PROMPT),
            TechType.FEW: gen(FEW_SHOT_PROMPT),
            TechType.COT: gen(COT_PROMPT),
            TechType.REACT: gen(REACT_PROMPT)
        }

        return techies_rsg