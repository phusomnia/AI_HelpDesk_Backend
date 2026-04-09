from fastapi import APIRouter, FastAPI, status, Depends
from SharedKernel.base.APIResponse import APIResponse
from SharedKernel.persistence.Decorators import Controller
from SharedKernel.utils.yamlenv import load_env_yaml
from sqlmodel.ext.asyncio.session import AsyncSession
from sqlmodel import text
from SharedKernel.persistence.PersistenceManager import get_db_session
from SharedKernel.persistence.Neo4jManager import get_neo4j_manager, Neo4jManager
from SharedKernel.config.LLMConfig import LLMFactory, EmbeddingFactory
config = load_env_yaml()

@Controller
class SharedKernelController:
    def __init__(self, app: FastAPI) -> None:
        self.app = app
        self.router = APIRouter(prefix="/shared_kernel", tags=["Shared Kernel"])
        self.register_route()
        self.app.include_router(self.router)

    def register_route(self):
        @self.router.get("/db")
        async def check_db_health(session: AsyncSession = Depends(get_db_session)):
            try:
                await session.exec(text("SELECT 1"))
                return APIResponse(
                    message="Database is healthy",
                    status_code=status.HTTP_200_OK,
                    data={"status": "healthy"},
                )
            except Exception as e:
                return APIResponse(
                    message="Database is unhealthy",
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    data={"status": "unhealthy", "error": str(e)},
                )

        @self.router.get("/neo4j")
        def check_neo4j_health(neo4j_mgr: Neo4jManager = Depends(get_neo4j_manager)):
            try:
                is_healthy = neo4j_mgr.verify_connectivity()
                if is_healthy:
                    return APIResponse(
                        message="Neo4j is healthy",
                        status_code=status.HTTP_200_OK,
                        data={"status": "healthy"},
                    )
                else:
                    return APIResponse(
                        message="Neo4j is unhealthy",
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        data={"status": "unhealthy", "error": "Connection failed"},
                    )
            except Exception as e:
                return APIResponse(
                    message="Neo4j is unhealthy",
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    data={"status": "unhealthy", "error": str(e)},
                )

        @self.router.get("/llm")
        async def check_llm_health():
            try:
                provider_name = config.llm.provider
                llm = LLMFactory.create(provider_name)

                response = llm.invoke("hello")

                return APIResponse(
                    message="LLM is healthy",
                    status_code=status.HTTP_200_OK,
                    data={
                        "status": "healthy",
                        "provider": provider_name,
                        "model": config.llm.get(provider_name, {}).get("model")
                        if isinstance(config.llm, dict)
                        else getattr(config.llm, provider_name, {}).model
                        if hasattr(config.llm, provider_name)
                        else config.llm.ollama.model,
                        "response": response.content[:100]
                        if hasattr(response, "content")
                        else str(response)[:100],
                    },
                )
            except Exception as e:
                return APIResponse(
                    message="LLM is unhealthy",
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    data={"status": "unhealthy", "error": str(e)},
                )

        @self.router.get("/embedding")
        async def check_embedding_health():
            try:
                provider_name = config.llm.provider
                embedding = EmbeddingFactory.create(provider_name)

                vector = embedding.embed_query("test")

                return APIResponse(
                    message="Embedding is healthy",
                    status_code=status.HTTP_200_OK,
                    data={
                        "status": "healthy",
                        "provider": provider_name,
                        "model": getattr(getattr(config.llm, provider_name, None), 'embed', 'unknown')
                        if hasattr(config.llm, provider_name)
                        else "unknown",
                        "dimension": len(vector),
                    },
                )
            except Exception as e:
                return APIResponse(
                    message="Embedding is unhealthy",
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    data={"status": "unhealthy", "error": str(e)},
                )
