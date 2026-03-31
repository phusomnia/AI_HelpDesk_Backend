import datetime
import json
from typing import List
import uuid
from typing import List
from fastapi import Depends, UploadFile
import uuid6
from SharedKernel.base.Logger import get_logger
from src.Domain.base_entities import Tickets
from src.Features.TicketAPI.TicketRepository import TicketRepository
from src.Features.TicketAPI.TicketDTO import TicketBaseDTO, TicketSearchRequest
from src.Features.RealTimeAPI.Storage.StorageService import StorageService

logger = get_logger(__name__)
new_id = uuid6.uuid7()

class TicketService:
    def __init__(
        self, 
        ticket_repo: TicketRepository = Depends(),
        file_service: StorageService = Depends(),
    ):
        self.ticket_repo = ticket_repo
        self.file_service = file_service

    async def search(self, req: TicketSearchRequest):
        return await self.ticket_repo.search_tickets(req)

    async def create_ticket(self, dto: TicketBaseDTO):
        model = Tickets()
        ticket = self.ticket_repo.update_model_from_dto(model, dto)
        return await self.ticket_repo.save(ticket)

    async def create_ticket_with_attachments(self, 
        dto: TicketBaseDTO, 
        files: List[UploadFile]
    ):
        model = Tickets()

        if files:
            result = await self.file_service.save_files(files)
            model.attachment_url = [{"url": url} for url in result.get("urls", [])]
        else:
            model.attachment_url = []

        ticket = self.ticket_repo.update_model_from_dto(model, dto)
        return await self.ticket_repo.save(ticket)
    
    # 
    # async def edit_ticket_with_attachments(self, id: str, dto: TicketUpdateDTO, files: List[UploadFile]) -> Tickets:
    #     logger.info(f"DTO: {dto}")
    #     model = await self.ticket_repo.find_by_id(uuid.UUID(id))
    #     dto.attachment_url = json.loads(dto.attachment_url)
        
    #     if files:
    #         logger.info("Files...")
    #         existing_attachments = []
    #         if dto.attachment_url:
    #             try:
    #                 if isinstance(dto.attachment_url, str):
    #                     existing_attachments = json.loads(dto.attachment_url)
    #                 elif isinstance(dto.attachment_url, list):
    #                     existing_attachments = dto.attachment_url
    #             except:
    #                 existing_attachments = []

    #         for file in files: 
    #             file.filename = f"{new_id}_{file.filename}"
    #             await self.file_service.save_file(file)
    #             format_result = { "url": f"http://localhost:8080/api/v1/files/{file.filename}" }
    #             existing_attachments.append(format_result)

    #         dto.attachment_url = existing_attachments
        
    #     ticket = self.ticket_repo.update_model_from_dto(model, dto)
    #     logger.info(f"Edit ticket: {ticket}")
    #     return await self.ticket_repo.save(ticket)

    async def soft_delete_ticket(self, id: str):
        model = await self.ticket_repo.find_by_id(uuid.UUID(id))
        model.delete_at = datetime.datetime.now()
        if not model:
            raise ValueError(f"Ticket with ID {id} not found")
        return await self.ticket_repo.save(model)
    

    # async def create_raw(self):
    #     query = """
    #         INSERT INTO Ticket (subject) VALUES ('abc');
    #     """

    #     return await self.ticket_repo.execute_raw(query)

    async def update_ticket(self, ticket_id: str, dto: TicketBaseDTO) -> Tickets:
        ticket = await self.ticket_repo.find_by_id(ticket_id)
        
        for field, value in dto.model_dump(exclude_unset=True).items():
            if hasattr(ticket, field):
                setattr(ticket, field, value)
        
        return await self.ticket_repo.update(ticket)

    # Static 
    async def get_status_statistics(self):
        """Thống kê số lượng ticket theo status"""
        return await self.ticket_repo.get_status_statistics()

    async def get_priority_statistics(self):
        """Thống kê số lượng ticket theo priority"""
        return await self.ticket_repo.get_priority_statistics()

    async def get_time_statistics(self, year: int = None, month: int = None):
        """Thống kê ticket theo thời gian (tháng/năm)"""
        return await self.ticket_repo.get_time_statistics(year, month)
