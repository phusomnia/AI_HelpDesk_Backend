import datetime
from fastapi import Depends
from src.Features.DeparmentAPI.DeparmentRepository import DepartmentRepository
from src.Features.DeparmentAPI.DepartmentDTO import DepartmentSearchRequest, DepartmentCreateDTO, DepartmentUpdateDTO
from src.Domain.base_entities import Departments
from starlette import status
from SharedKernel.exception.APIException import APIException

class DepartmentService:
    def __init__(self, repo: DepartmentRepository = Depends()):
        self.repo = repo

    async def search_departments(self, req: DepartmentSearchRequest):
        return await self.repo.search_department(req)

    async def create_department(self, dto: DepartmentCreateDTO):
        department = Departments()
        department.name = dto.name
        return await self.repo.save(department)

    async def edit_department(self, id: str, dto: DepartmentUpdateDTO):
        department = await self.repo.find_by_id(id)
        if not department:
            raise APIException(
                message="Department not found",
                status_code=status.HTTP_404_NOT_FOUND
            )
        
        if dto.name:
            department.name = dto.name
        
        return await self.repo.update(department)

    async def delete_department(self, id: str):
        department = await self.repo.find_by_id(id)
        if not department:
            raise APIException(
                message="Department not found",
                status_code=status.HTTP_404_NOT_FOUND
            )
        
        department.delete_at = datetime.datetime.now()
        return await self.repo.save(department)