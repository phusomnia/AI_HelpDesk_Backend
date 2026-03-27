from langchain_core.messages import HumanMessage, SystemMessage
from Features.LangChainAPI.LangChainDTO import Callback, ChatRequest, ChatTechniqueRequest, ChatTemplateRequest, TechType, TemplateType
from SharedKernel.persistence.Decorators import Service
from SharedKernel.utils.yamlenv import load_env_yaml
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import BaseModel, Field, config
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser

from src.Features.LangChainAPI.prompt import System_Instruction

class YouTubeVideo(BaseModel):
    title: str = Field(description="Tiêu đề video")
    channel: str = Field(description="Tên kênh YouTube")
    views: int = Field(description="Số lượt xem")
    upload_date: str = Field(description="Ngày đăng video")
    is_short: bool = Field(description="Video có phải YouTube Shorts hay không")

class PromptService:
    def __init__(self, provider: BaseChatModel, callbacks: Callback):
        self.provider = provider
        self.callbacks = callbacks 

    async def aprompt(self, req: ChatRequest):
        return await self.callbacks.ainvoke(self.provider, [System_Instruction(req)])
        pass

    async def asprompt(self, req: ChatRequest) -> dict:
        return await self.callbacks.astream(self.provider, [System_Instruction(req)])

    async def GameAwardQA(self, question: str):
        GAME_AWARDS_2026 = {
            "Game of the Year": "Clair Obscur: Expedition 33",
            "Best RPG": "Clair Obscur: Expedition 33",
            "Best Action Game": "Hades II",
            "Best Adventure Game": "Ghost of Yōtei",
            "Best Strategy / Simulation": "The Alters",
            "Best Racing Game": "Mario Kart World",
            "Best Online Game": "ARC Raiders",
            "Best Sports Game": "Rematch"
        }

        def build_context():
            context = "Game Awards 2026 winners:\n"

            for category, game in GAME_AWARDS_2026.items():
                context += f"{category}: {game}\n"

            return context
        print(build_context())

        messages = [SystemMessage(content=f"""
        Bạn là trợ lý hỏi đáp về Game Awards.

        Sử dụng dữ liệu sau để trả lời:

        {build_context()}

        Trả lời ngắn gọn.
        Nếu không có dữ liệu thì nói "Không có thông tin".
        """)]

        messages.append(HumanMessage(content=question))
        stream = self.provider.astream(messages)

        async for chunk in stream:
            yield chunk.content
        ...

    async def create_blog_with_ai(self, title: str) -> dict:
        template = "Viết một blog ngắn nói về {topic}" 
        _prompt = PromptTemplate.from_template(template)
        prompt = _prompt.format(topic=title)

        return await self.callbacks.astream(self.provider, prompt)

    async def extract_youtube_video_info(
        self, 
        description: str):
        instruction = """
        Trích xuất thông tin video YouTube từ đoạn sau.

        {format_instructions}

        Description:
        {description}
        """

        parser = PydanticOutputParser(pydantic_object=YouTubeVideo)

        template = PromptTemplate(
            template=instruction,
            input_variables=["description"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        chain = template | self.provider | parser
        stream = chain.astream({"description": description})
        
        async for chunk in stream:
            yield chunk.model_dump_json()

        
