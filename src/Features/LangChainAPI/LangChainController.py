from typing import List
from Features.LangChainAPI.LangChainFacade import LangChainFacade
from Features.LangChainAPI.LangTools import LangTools
from SharedKernel.ai.AIConfig import AIConfigFactory
from fastapi import APIRouter, FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from lagom import Container
from Features.LangChainAPI.LangChainDTO import ChatMessageRequest, ChatRequest, ChatTechniqueRequest, ChatTemplateRequest, PromptType, TechType, TemplateType
from SharedKernel.persistence.Decorators import Controller

@Controller
class LangChainController:
    def __init__(self, app: FastAPI, container: Container) -> None:
        self.app = app
        self.container = container 
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
        # self.container[AIConfigFactory] = AIConfigFactory
        langfacade = self.container[LangChainFacade]

        @self.router.post("/")
        async def prompt(
            req: ChatRequest,
            prompt_type: PromptType = PromptType.NONE
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

        @self.router.get("/technique")
        async def prompt_technique(
            tech_type: TechType
        ):
            dto = ChatTechniqueRequest(message="", tech=tech_type)
            handler = await langfacade.prompt.promt_techniques(dto)
            result = await handler.get(tech_type)

            if result is None:
                raise ValueError(f"Invalid prompt type: {tech_type}")

            return StreamingResponse(
                result["content"],
                media_type="text/event-stream"
            )
        ...

        @self.router.post("/template")
        async def prompt_template(
            req: ChatRequest,
            template_type: TemplateType
        ):
            dto = ChatTemplateRequest(message=req.message, template=template_type)
            handler = await langfacade.prompt.prompt_template(dto)
            result = await handler.get(template_type)
            
            if handler is None:
                raise ValueError(f"Invalid prompt type: {template_type}")   

            return StreamingResponse(
                result["content"],
                media_type="text/event-stream"
            )
        ...

        @self.router.post("/structed_output")
        async def structed_output():
            result = await langfacade.output_parser.structed_output()
            return result
        ...

        @self.router.post("/tools")
        async def tools(
            req: ChatRequest
        ):
            handler = await langfacade.tools.call_search_tools()
            result = handler.get("content")

            return StreamingResponse(
                result,
                media_type="text/event-stream"
            )

        @self.router.post("/short_chat")
        async def short_chat(
            req: ChatMessageRequest
        ):
            return await langfacade.memory.short_chat(req)

        @self.router.post("/long_chat")
        async def long_chat(
            req: ChatMessageRequest
        ):
            print(req.session_id)
            return await langfacade.memory.long_chat(req)

        @self.router.post("/document_webpage")
        def document_webpage(
            req: ChatRequest
        ):
            result = langfacade.SYN.loader.load_webpage(req.message)
 
            return StreamingResponse(
                result,
                media_type="text/event-stream"
            )
        ...

        @self.router.post("/document_pdf")
        async def document_pdf(
            files: List[UploadFile] = File(...),
        ):
            for file in files:
                await langfacade.SYN.ingest_pdf(file)

        @self.router.post("/search_document")
        async def search_document(query: str):
            result = await langfacade.SYN.call_rag(query)
            return StreamingResponse(
                result["answer"],
                media_type="text/event-stream"
            )
            ...
            
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