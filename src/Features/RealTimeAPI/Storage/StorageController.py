from enum import Enum
from typing import List
from fastapi import APIRouter, Depends, FastAPI, File, Query, Request, UploadFile, status 
from fastapi.responses import FileResponse
from SharedKernel.persistence.Decorators import Controller
from src.Features.RealTimeAPI.Storage.StorageService import StorageService
from src.Features.RealTimeAPI.Storage.FileDTO import FileSearchRequest, TypeStorage
from src.SharedKernel.base.APIResponse import APIResponse

@Controller
class StorageController:
    def __init__(self, app: FastAPI) -> None:
        self.app = app
        self.storage_service = StorageService()
        self.router = APIRouter(
            prefix="/api/v1/storage",
            tags=["Storage"]
        )
        self.register_route()
        self.app.include_router(self.router)

    def register_route(self):

        @self.router.get("/files")
        async def get_all_files(req: FileSearchRequest = Depends(), storage_service: StorageService = Depends()):
            result = await storage_service.get_all_files(req)
            
            return APIResponse(
                message="Get all files successfully",
                status_code=status.HTTP_200_OK,
                data=result
            )

        @self.router.get("/files/{file_id}/{file_name}")
        async def get_file_by_id(
            file_id: str,
            file_name: str,
            storage_service: StorageService = Depends()
        ):
            result = await storage_service.get_file_by_id(file_id, file_name)

            response = FileResponse(
                path=result["file_path"],
                filename=file_name,
                media_type=result["mime_type"] or "application/octet-stream", 
                content_disposition_type="inline"
            )

            response.headers["Access-Control-Allow-Origin"] = "*"
            response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
            response.headers["Access-Control-Allow-Headers"] = "*"
            
            return response
            
        @self.router.post("/files/upload")
        async def upload_file(
            files: List[UploadFile] = File(...),
            storage_service: StorageService = Depends()
        ):
            response = await storage_service.save_files(files)

            return APIResponse(
                message="Upload file successfully",
                status_code=status.HTTP_200_OK,
                data=response
            )

        @self.router.delete("/files/{file_id}")
        async def delete_file(
            file_id: str,
            storage_service: StorageService = Depends()
        ):
            await storage_service.delete_file(file_id)

            return APIResponse(
                message="Delete file successfully",
                status_code=status.HTTP_200_OK,
                data=None
            )