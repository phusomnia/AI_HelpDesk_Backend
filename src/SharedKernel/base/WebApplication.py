from dataclasses import dataclass
from functools import wraps
import importlib
import inspect
import json
import os
import pkgutil
import traceback
from typing import List, Type
from fastapi.responses import JSONResponse
from sqlmodel.ext.asyncio.session import AsyncSession
from fastapi import FastAPI
from sqlmodel import text
from SharedKernel.base.Logger import get_logger
from SharedKernel.persistence.PersistenceManager import PersistenceManagerFactory
from SharedKernel.socket.SocketManager import SocketManager
from scalar_fastapi import get_scalar_api_reference
from SharedKernel.utils.yamlenv import load_env_yaml
from fastapi.middleware.cors import CORSMiddleware
from src.SharedKernel.exception.APIException import APIException

config = load_env_yaml()
print(config.openapi.litestar.url)

logger = get_logger(__name__)

class WebApplication(FastAPI):
    def __init__(self, **kwargs):
        kwargs.setdefault("title", "API")
        kwargs.setdefault("docs_url", None)
        kwargs.setdefault("redoc_url", None)
        super().__init__(**kwargs)

        self.app_router()
        self.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"]
        )
    def app_router(self):
        @self.get("/hello", tags=["Hello"])
        async def hello():
            return {"message": "Hello, World!", "from": "WebApplication"}

        @self.get("/scalar", include_in_schema=False)
        async def scalar_docs():
            return get_scalar_api_reference(
                openapi_url=self.openapi_url,
                title=self.title,
            )

        @self.get("/check_db_health")
        async def check_db_health():
            try:
                async with self.di_container[AsyncSession] as session:
                    await session.execute(text("SELECT 1"))
                return {"status": "healthy"}
            except Exception as e:
                return {"status": "unhealthy", "error": str(e)}

        @self.exception_handler(APIException)
        async def api_exception_handler(request, exc: APIException):
            return JSONResponse(
                status_code=exc.status_code,
                content={
                    "message": exc.message,
                    "status_code": exc.status_code
                }
            )
    

    def scan_controllers(self, base_path: str = "src.Features") -> List[Type]:
        """
        Quét controllers dùng pkgutil như DIContainer
        """
        controllers = []
        
        try:
            package = importlib.import_module(base_path)
            logger.info(f"[WebApp] Package: {package}")

            for _, module_name, is_pkg in pkgutil.walk_packages(
                package.__path__, package.__name__ + "."
            ):
                if "Controller" in module_name:
                    try:
                        module = importlib.import_module(module_name)
                        logger.info(f"[WebApp] Imported: {module_name}")

                        for name, obj in inspect.getmembers(module, inspect.isclass):
                            if hasattr(obj, '__di_type__') and obj.__di_type__ == "controller":
                                controllers.append(obj)
                                logger.info(f"[WebApp] Found controller: {name}")
                                
                    except Exception as e:
                        logger.error(f"[WebApp] Error importing {module_name}: {e}")
                        logger.error(f"[WebApp] Full traceback: {traceback.format_exc()}")
                        
        except Exception as e:
            logger.info(f"[WebApp] Error scanning package {base_path}: {e}")
        
        return controllers
    
    def auto_register_controllers(self):
        """
        Tự động đăng ký tất cả controllers
        """
        controllers = self.scan_controllers()
        
        for controller_class in controllers:
            try:
                controller_class(self)
                logger.info(f"Registered controller: {controller_class.__name__}")
            except Exception as e:
                logger.info(f"Error registering {controller_class.__name__}: {e}")

    def map_controller(self):
        self.auto_register_controllers()
        pass

    

    