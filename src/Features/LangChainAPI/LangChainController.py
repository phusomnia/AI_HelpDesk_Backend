from typing import List, Optional

from pydantic import BaseModel
from Features.LangChainAPI.LangChainFacade import LangChainFacade
from Features.LangChainAPI.LangTools import LangTools
from SharedKernel.config.LLMConfig import LLMFactory
from fastapi import APIRouter, Depends, FastAPI, File, UploadFile, status
from fastapi.responses import StreamingResponse
from Features.LangChainAPI.LangChainDTO import ChatRequest
from SharedKernel.persistence.Decorators import Controller
from src.SharedKernel.base.APIResponse import APIResponse


@Controller
class LangChainController:
    def __init__(self, app: FastAPI) -> None:
        self.app = app
        self.router = APIRouter(prefix="/api/v1/langchain", tags=["LangChain"])
        self.tool_router = APIRouter(prefix="/api/v1/tools", tags=["Tools"])
        self.tools = LangTools()
        self.register_route()
        self.app.include_router(self.router)

    def register_route(self):
        #
        # RAG
        #
        @self.router.get("/chat_history/{session_id}")
        async def get_chat_history(
            session_id: str, langfacade: LangChainFacade = Depends()
        ):
            """Get paginated chat history for a session"""
            response = await langfacade.synthesizer.memory_repo.get_history_all(
                session_id=session_id,
            )

            return APIResponse(
                message="Chat history retrieved successfully",
                status_code=status.HTTP_200_OK,
                data=response,
            )

        @self.router.post("/load_document_pdf_PaC")
        async def load_document_pdf_PaC(
            files: List[UploadFile] = File(...), 
            langfacade: LangChainFacade = Depends()
        ):
            for file in files:
                await langfacade.synthesizer.ingest_file_PaC(file)

            return APIResponse(
                message=f"Successfully processed {len(files)} PDF file(s)",
                status_code=status.HTTP_200_OK,
                data=None,
            )

        class DeleteDocumentRequest(BaseModel):
            filename: str

        @self.router.delete("/delete_document")
        async def delete_document(
            req: DeleteDocumentRequest, 
            langfacade: LangChainFacade = Depends()
        ):
            await langfacade.synthesizer.delete_document_by_file_name(req.filename)
            return APIResponse(
                message=f"Delete successfully",
                status_code=status.HTTP_200_OK,
                data=None,
            )
            ...

        class RetrieveDocumentRequest(BaseModel):
            query: str
            session_id: str

        @self.router.post("/retrieve_document")
        async def retrieve_document(
            req: RetrieveDocumentRequest, 
            langfacade: LangChainFacade = Depends()
        ):
            return StreamingResponse(
                await langfacade.synthesizer.retriver_documents_PaC(req.query, req.session_id),
                media_type="text/event-stream",
            )
            ...

        #
        # GRAPH
        # 
        @self.router.post("/build-graph")
        async def build_graph(
            file: UploadFile = File(...), 
            langfacade: LangChainFacade = Depends()
        ):
            source = file.filename
            result = await langfacade.synthesizer.build_graph(file, source)
            return APIResponse(
                message="Graph built successfully",
                status_code=status.HTTP_200_OK,
                data=result,
            )

        @self.router.get("/graph/{source}/stats")
        async def get_graph_stats(
            source: str, 
            langfacade: LangChainFacade = Depends()
        ):
            stats = await langfacade.synthesizer.neo4j_store.get_graph_stats(source)
            return APIResponse(
                message="Graph stats retrieved",
                status_code=status.HTTP_200_OK,
                data=stats,
            )

        class GraphSearchRequest(BaseModel):
            query: str
            top_k: int = 5

        @self.router.post("/graph/search")
        async def search_graph(
            req: GraphSearchRequest, 
            langfacade: LangChainFacade = Depends()
        ):
            results = await langfacade.synthesizer.neo4j_store.search_by_embedding(
                req.query, req.top_k
            )
            return APIResponse(
                message="Search completed",
                status_code=status.HTTP_200_OK,
                data={"results": results},
            )

        class GraphQueryRequest(BaseModel):
            query: str
            source: Optional[str] = None

        @self.router.post("/graph/query")
        async def query_graph(
            req: GraphQueryRequest, 
            langfacade: LangChainFacade = Depends()
        ):
            result = await langfacade.synthesizer.query_graph_rag(req.query, req.source)
            return APIResponse(
                message="Query completed", status_code=status.HTTP_200_OK, data=result
            )

        @self.router.delete("/graph/{source}")
        async def delete_graph(
            source: str, 
            langfacade: LangChainFacade = Depends()
        ):
            await langfacade.synthesizer.neo4j_store.delete_graph(source)
            return APIResponse(
                message="Graph deleted successfully",
                status_code=status.HTTP_200_OK,
                data=None,
            )

        ...

        #
        # TOOL 
        #
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
                self.tools.ascrawl_web(req.message), media_type="text/event-stream"
            )

    ...
