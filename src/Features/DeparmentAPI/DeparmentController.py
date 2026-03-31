from fastapi import APIRouter, FastAPI, status
from fastapi.params import Depends
from Features.DeparmentAPI.DepartmentDTO import DepartmentSearchRequest, DepartmentCreateDTO, DepartmentUpdateDTO
from SharedKernel.base.APIResponse import APIResponse
from SharedKernel.persistence.Decorators import Controller
from src.Features.DeparmentAPI.DepartmentService import DepartmentService

@Controller
class DepartmentController:
    def __init__(self, app: FastAPI) -> None:
        self.app = app
        self.router = APIRouter(
            prefix="/api/v1/departments",
            tags=["Department"]
        )
        self.register_route()
        self.app.include_router(self.router)

    def register_route(self):
        @self.router.get("/")
        async def get_departments(
            query_dto: DepartmentSearchRequest = Depends(), 
            department_service: DepartmentService = Depends()
        ):
            result = await department_service.search_departments(query_dto)
            return APIResponse(
                message="Get departments",
                status_code=status.HTTP_200_OK,
                data=result
            )

        @self.router.post("/", response_model=None)
        async def create_department(
            dto: DepartmentCreateDTO, 
            department_service: DepartmentService = Depends()
        ):
            result = await department_service.create_department(dto)
            return APIResponse(
                message="Create department",
                status_code=status.HTTP_201_CREATED,
                data=result
            )

        @self.router.put("/{id}", response_model=None)
        async def edit_department(
            id: str,
            dto: DepartmentUpdateDTO,
            department_service: DepartmentService = Depends()
        ):
            result = await department_service.edit_department(id, dto)
            return APIResponse(
                message="Edit department",
                status_code=status.HTTP_200_OK,
                data=result
            )

        @self.router.delete("/{id}")
        async def delete_department(
            id: str,
            department_service: DepartmentService = Depends()
        ):
            result = await department_service.delete_department(id)
            return APIResponse(
                message="Delete department",
                status_code=status.HTTP_200_OK,
                data=result
            )
