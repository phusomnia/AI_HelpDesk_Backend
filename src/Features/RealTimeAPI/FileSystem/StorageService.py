import datetime
import glob
from io import BytesIO
import mimetypes
import os
import shutil
from typing import List
import uuid6
from fastapi import Depends, UploadFile, status
from src.Features.LangChainAPI.LangChainFacade import LangChainFacade
from src.Domain.base_entities import Attachment
from SharedKernel.exception.APIException import APIException
from SharedKernel.base.Logger import get_logger
from src.Features.RealTimeAPI.FileSystem.StorageRepository import FileRepository
from src.Features.RealTimeAPI.FileSystem.FileDTO import FileSearchRequest
from src.SharedKernel.utils.yamlenv import load_env_yaml

class StorageService:
    def __init__(self, 
        repo: FileRepository = Depends(),
        langfacade: LangChainFacade = Depends()
    ):
        self.log = get_logger(__name__)
        self.config = load_env_yaml()
        self.STORAGE_PREFIX  = "api/v1/storage/files" 
        self.repo = repo
        self.langfacade = langfacade

    async def get_all_files(self, req: FileSearchRequest):
        return await self.repo.search_files(req)
        pass
    
    async def save_files(self, files: List[UploadFile]):
        attachment_urls = []
        overwritten_files = []
        
        for file in files:
            # Check existing file by name
            existing_file = await self.repo.find_by_filename(file.filename)
            print(existing_file)

            if existing_file:
                self.log.info(f"Found existing file: {file.filename}, deleting...")
                
                # Delete from filesystem
                old_file_path = self._get_file_path(str(existing_file['id']), existing_file['file_name'])
                if os.path.exists(old_file_path):
                    os.remove(old_file_path)
                    self.log.info(f"Deleted file from disk: {old_file_path}")
                
                # Soft delete from database
                existing_file['delete_at'] = datetime.datetime.now()
                await self.repo.soft_delete_by_filename(file.filename)
                overwritten_files.append(file.filename)
            
            # Save new file
            attachment = Attachment()
            new_id = uuid6.uuid7()
            file_name = os.path.splitext(file.filename)[0]
            file_ext = os.path.splitext(file.filename)[1]
            filename = f"{file_name}{file_ext}"
            
            file_path = self._get_file_path(str(new_id), filename)

            content = await file.read()
            with open(file_path, "wb") as f:
                f.write(content)

            if file_ext.lower() == ".pdf":
                # Reset con trỏ file về đầu để ingest
                await file.seek(0)
                await self.langfacade.synthesizer.ingest_file_PaC(file)

            attachment.id = new_id
            attachment.file_name = filename
            attachment.url = "http://{}:{}/{}/{}/{}".format(
                self.config.app.host, 
                self.config.app.port,
                f"{self.STORAGE_PREFIX}",
                new_id,
                filename,
            )
            await self.repo.save(attachment)
            attachment_urls.append(attachment.url)

        return {
            "urls": attachment_urls,
            "overwritten": overwritten_files
        }
            
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

        await self.langfacade.synthesizer.delete_document_by_file_name(file.file_name)

    def _get_file_path(self, folder_storage: str, filename: str):
        full_path = os.path.join("./static", folder_storage, filename)
        dir_path = os.path.dirname(full_path)
        
        if not os.path.exists(dir_path):
            os.makedirs(dir_path, exist_ok=True)
        
        return full_path
