from typing import List

from pydantic import BaseModel
from Features.LangChainAPI.LangChainFacade import LangChainFacade
from Features.LangChainAPI.LangTools import LangTools
from SharedKernel.ai.AIConfig import AIConfigFactory
from fastapi import APIRouter, Depends, FastAPI, File, Query, UploadFile
from fastapi.responses import StreamingResponse
from lagom import Container
from Features.LangChainAPI.LangChainDTO import ChatMessageRequest, ChatRequest, ChatTechniqueRequest, ChatTemplateRequest, PromptType, RagRequest, RagType, TechType, TemplateType
from SharedKernel.persistence.Decorators import Controller

@Controller
class BaiTapController:
    def __init__(self, app: FastAPI) -> None:
        """
        Moi chuong tuong ung voi tuan hoc
        """
        
        self.app = app
        self.router = APIRouter(
            prefix="/api/v1/bai_tap",
            tags=["BTCNLTHD"]
        )
        self.tools = LangTools()
        self.Chuong_1()
        self.Chuong_2()
        self.Chuong_3()
        self.Chuong_4()
        self.Chuong_5()
        self.Chuong_6()
        self.Chuong_7()
        self.app.include_router(self.router)

    def Chuong_1(self):
        """
        Ung dung nhan tin voi AI
        """

        @self.router.post("/chat_with_ai")
        async def chat_with_ai(
            req: ChatRequest,
            prompt_type: PromptType = PromptType.NONE,
            langfacade: LangChainFacade = Depends()
        ):
            async def handle_none(req):
                return await langfacade.prompt.aprompt(req)

            async def handle_stream(req):
                result = await langfacade.prompt.asprompt(req)
                return StreamingResponse(
                    result["content"],
                    media_type="text/event-stream"
                )

            prompt_dict = {
                PromptType.NONE: handle_none,
                PromptType.STREAM: handle_stream
            }

            handler = prompt_dict.get(prompt_type)
            if handler is None:
                raise ValueError(f"Invalid prompt type: {prompt_type}")
            return await handler(req)
        ...

    def Chuong_2(self):
        """
        Ung dung hoi dap game award
        """
        class ChatRequest(BaseModel):
            question: str="Game of the Year 2026 là gì?"
        @self.router.post("/game_awards_QA")
        async def game_awards_QA(req: ChatRequest, langfacade: LangChainFacade = Depends()):
            return StreamingResponse(
                langfacade.prompt.GameAwardQA(req.question),
                media_type="text/event-stream"
            )
        ...

    def Chuong_3(self):
        """
        Ung dung tao blog voi AI
        """

        class BlogRequest(BaseModel):
            title: str
        @self.router.post("/create_blog_with_ai")
        async def create_blog_with_ai(
            req: BlogRequest, 
            langfacade: LangChainFacade = Depends()
        ):
            response = await langfacade.prompt.create_blog_with_ai(req.title)

            return StreamingResponse(
                response["content"],
                media_type="text/event-stream"
            )
        ...

    def Chuong_4(self):
        """
        Ung dung format dinh dang thong tin video 
        """

        class DescriptionVideo(BaseModel):
            description: str = """Video 'LangChain Tutorial for Beginners' của kênh AI Academy, 120000 views, đăng ngày 2024-05-12. Đây không phải Shorts."""
        @self.router.post("/format_video")
        async def format_video(
            req: DescriptionVideo,
            langfacade: LangChainFacade = Depends() 
        ):
            return StreamingResponse(
                langfacade.prompt.extract_youtube_video_info(req.description),
                media_type="text/event-stream"
            )
        ...

    def Chuong_5(self):
        """
        Ung dung viet tieu thuyet
        Query: Cô bạn bàn bên
        """

        class NovelAgentRequest(BaseModel):
            description: str

        @self.router.post("/novel_agent")
        async def novel_agent(
            req: NovelAgentRequest,
            langfacade: LangChainFacade = Depends()    
        ):
            response = langfacade.agent.write_narrative(req.description)
            return StreamingResponse(
                response,
                media_type="text/event-stream"
            )
        ...

    def Chuong_6(self):
        """
        Ung dung ghi chu 

        query: Ngày 12/6 tôi sẽ bay về quê để dự đám cưới
        """

        class TakeNoteRequest(BaseModel):
            query: str
            session_id: str = "user_1"

        @self.router.post("/take_note")
        async def take_note(
            req: TakeNoteRequest,
            langfacade: LangChainFacade = Depends()
        ):
            response = await langfacade.agent.take_note(req.session_id, req.query)
            return StreamingResponse(
                response,
                media_type="text/event-stream"
            )
        ...

    def Chuong_7(self):
        """
        Ung dung tim kiem da linh vuc

        query: Danh sách oscar 2026
        """

        class MulitDomainRequest(BaseModel):
            query: str

        @self.router.post("/search_multi_domain")
        async def search_multi_domain(
            req: MulitDomainRequest,
            langfacade: LangChainFacade = Depends()
        ):
            response = langfacade.agent.search_multi_domain(req.query)
            return StreamingResponse(
                response,
                media_type="text/event-stream"
            )
        ...
    ...

