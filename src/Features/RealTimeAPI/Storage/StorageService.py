import datetime
import glob
import mimetypes
import os
import shutil
from typing import List
import uuid6
from fastapi import Depends, UploadFile, status
from src.Features.RealTimeAPI.Storage.FileDTO import FileSearchRequest, TypeStorage
from src.Domain.base_entities import Attachment
from SharedKernel.exception.APIException import APIException
from SharedKernel.base.Logger import get_logger
from src.Features.RealTimeAPI.Storage.StorageRepository import FileRepository
from src.SharedKernel.persistence.Decorators import Transaction
from src.SharedKernel.utils.yamlenv import load_env_yaml

logger = get_logger(__name__)
config = load_env_yaml()

class StorageService:
    def __init__(self, 
        repo: FileRepository = Depends()
    ):
        self.STORAGE_PREFIX  = "api/v1/storage/files" 
        self.repo = repo

    async def get_all_files(self, req: FileSearchRequest):
        return await self.repo.search_files(req)
        pass
    
    async def save_files(self,
        files: List[UploadFile]
    ):
        attachment_urls = []
        for file in files:
            attachment = Attachment()

            new_id = uuid6.uuid7()
            file_name = os.path.splitext(file.filename)[0]
            file_ext = os.path.splitext(file.filename)[1]
            filename = f"{file_name}{file_ext}"
            
            file_path = self._get_file_path(str(new_id), filename)

            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)

            attachment.id = new_id
            attachment.file_name = filename
            attachment.url = "http://{}:{}/{}/{}/{}".format(
                config.app.host, 
                config.app.port,
                f"{self.STORAGE_PREFIX}",
                new_id,
                filename,
            )
            await self.repo.save(attachment)
            attachment_urls.append(attachment.url)

        return {
            "urls": attachment_urls
        }
    ...
            
    async def get_file_by_id(self, file_id: str, file_name: str = None):
        file = await self.repo.find_by_id(file_id)
        if not file: 
            raise APIException("File not found", status_code=status.HTTP_404_NOT_FOUND)

        if file_name and file.file_name != file_name:
            raise APIException("File name mismatch", status_code=status.HTTP_400_BAD_REQUEST)
        file_name = file.file_name
        file_path = self._get_file_path(file_id, file_name)
        
        if not os.path.exists(file_path):
            raise APIException("Can't find file", status_code=status.HTTP_404_NOT_FOUND)
        mime_type, _ = mimetypes.guess_type(file_path)
        return {
            "file_path": file_path,
            "mime_type": mime_type
        }

    async def delete_file(self, file_id: str):
        file = await self.repo.find_by_id(file_id)
        
        if not file:
            raise APIException(
                "File not found in database",
                status_code=status.HTTP_404_NOT_FOUND
            )
        
        referencing_messages = await self.repo.fetch_all(
            """
            SELECT * 
            FROM Messages m 
            WHERE m.content LIKE :file_id_pattern
            AND m.delete_at IS NULL
            """,
            {"file_id_pattern": f"%{file_id}%"}
        )

        for message in referencing_messages:
            await self.repo.execute(
                """
                UPDATE Messages 
                SET delete_at = :delete_now 
                WHERE id = :message_id
                """,
                {
                    "delete_now": datetime.datetime.now(),
                    "message_id": message['id']
                }
            )
        
        folder_path = os.path.join("./static", file_id) 

        if os.path.exists(folder_path):
            shutil.rmtree(folder_path)

        file.delete_at = datetime.datetime.now()
        await self.repo.save(file)

    def _get_file_path(self, folder_storage: str, filename: str):
        full_path = os.path.join("./static", folder_storage, filename)
        dir_path = os.path.dirname(full_path)
        
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        
        return full_path
