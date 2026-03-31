import json
from typing import List
from fastapi import APIRouter, FastAPI, File, Form, UploadFile, status
from fastapi.params import Depends
from Features.TicketAPI.TicketDTO import TicketBaseDTO, TicketSearchRequest
from SharedKernel.base.APIResponse import APIResponse
from SharedKernel.persistence.Decorators import Controller
from SharedKernel.utils.yamlenv import load_env_yaml
from src.Features.TicketAPI.TicketService import TicketService

config = load_env_yaml()

@Controller
class TicketController:
    def __init__(self, app: FastAPI) -> None:
        self.app = app
        self.router = APIRouter(
            prefix="/api/v1/tickets",
            tags=["Ticket"]
        )
        self.register_route()
        self.statistic_route()
        self.app.include_router(self.router)

    def register_route(self):
        
        @self.router.get("/")
        async def get_ticket(
            query_dto: TicketSearchRequest = Depends(), 
            ticket_service: TicketService = Depends()
        ):
            result = await ticket_service.search(query_dto)

            return APIResponse(
                message="Get tickets",
                status_code=status.HTTP_200_OK,
                data=result
            )

        @self.router.post("/", response_model=None)
        async def create_ticket(
            ticket: str | None = Form(...),
            attachments: List[UploadFile] | None = File(None),
            ticket_service: TicketService = Depends()
        ):
            ticket_data = json.loads(ticket)
            dto = TicketBaseDTO(**ticket_data)
            print(dto)
            response = await ticket_service.create_ticket_with_attachments(dto, attachments)

            return APIResponse(
                message="Create ticket",
                status_code=status.HTTP_201_CREATED,
                data=response
            )

        @self.router.put("/{id}", response_model=None)
        async def edit_ticket(
            id: str,
            dto: TicketBaseDTO,
            ticket_service: TicketService = Depends()
        ):
            result = await ticket_service.update_ticket(id, dto)
            return APIResponse(
                message="Edit ticket",
                status_code=status.HTTP_200_OK,
                data=result
            )

    def statistic_route(self):
        @self.router.get("/dashboard/status")
        async def get_status_statistics(
            ticket_service: TicketService = Depends()
        ):
            result = await ticket_service.get_status_statistics()
            return APIResponse(
                message="Get status statistics",
                status_code=status.HTTP_200_OK,
                data=result
            )

        @self.router.get("/dashboard/priority")
        async def get_priority_statistics(
            ticket_service: TicketService = Depends()
        ):
            result = await ticket_service.get_priority_statistics()
            return APIResponse(
                message="Get priority statistics",
                status_code=status.HTTP_200_OK,
                data=result
            )

        @self.router.get("/dashboard/time")
        async def get_time_statistics(
            year: int = None,
            month: int = None,
            ticket_service: TicketService = Depends()
        ):
            result = await ticket_service.get_time_statistics(year, month)
            return APIResponse(
                message="Get time statistics",
                status_code=status.HTTP_200_OK,
                data=result
            )