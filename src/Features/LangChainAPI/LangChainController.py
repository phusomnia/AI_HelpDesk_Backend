from typing import List
from Features.LangChainAPI.LangChainFacade import LangChainFacade
from Features.LangChainAPI.LangTools import LangTools
from SharedKernel.ai.AIConfig import AIConfigFactory
from fastapi import APIRouter, Depends, FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from Features.LangChainAPI.LangChainDTO import ChatMessageRequest, ChatRequest, ChatTechniqueRequest, ChatTemplateRequest, PromptType, RagRequest, RagType, TechType, TemplateType
from SharedKernel.persistence.Decorators import Controller

@Controller
class LangChainController:
    def __init__(self, app: FastAPI) -> None:
        self.app = app
        self.router = APIRouter(
            prefix="/api/v1/langchain",
            tags=["LangChain"]
        )
        self.tool_router = APIRouter(
            prefix="/api/v1/tools",
            tags=["Tools"]
        )
        self.tools = LangTools()
        self.register_route()
        self.tool_route()
        self.app.include_router(self.router)
        self.app.include_router(self.tool_router)

    def register_route(self):

        @self.router.post("/chat")
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

        @self.router.post("/update_doc")
        async def load_document_pdf(
            files: List[UploadFile] = File(...),
            langfacade: LangChainFacade = Depends()
        ):
            for file in files:
                await langfacade.SYN.update_docs(file)

        @self.router.post("/load_document_pdf_PaC")
        async def load_document_pdf_PaC(
            files: List[UploadFile] = File(...),
            langfacade: LangChainFacade = Depends()
        ):
            for file in files:
                await langfacade.SYN.ingest_pdf_PaC(file)

        @self.router.post("/search_document")
        async def search_document(
            query: str, 
            rag_type: RagType,
            langfacade: LangChainFacade = Depends()
        ):
            req = RagRequest(query=query, rag_type=rag_type)
            response = await langfacade.SYN.call_rag(req)
            return StreamingResponse(
                response.content,
                media_type="text/event-stream"
            )
            ...

        @self.tool_router.post("/crawl")
        def web_crawler(url: str, langfacade: LangChainFacade = Depends()):
            langfacade.crawler.crawl_data(url)
            
    def tool_route(self):
        @self.tool_router.post("/tool_search")
        async def web_search(req: ChatRequest):
            urls = await self.tools.duckduckgo_search(req.message)
            contents = []
            for url in urls:
                content = await self.tools.crawl_web(url)
                contents.append(content)
            return contents

        @self.tool_router.post("/web_fetch")
        async def web_fetch(req: ChatRequest):
            content = await self.tools.crawl_web(req.message)
            return content

        @self.tool_router.post("/fetch")
        def web_asfetch(req: ChatRequest):
            # content = await self.tools.crawl(req.message)
            return StreamingResponse(
                self.tools.ascrawl_web(req.message),
                media_type="text/event-stream"
            )
        ...
    ...