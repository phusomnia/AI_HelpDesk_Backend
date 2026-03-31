import uuid
from fastapi import APIRouter, FastAPI, status
from fastapi.params import Depends
from sqlmodel.ext.asyncio.session import AsyncSession
from SharedKernel.base.APIResponse import APIResponse
from SharedKernel.persistence.Decorators import Controller
from SharedKernel.persistence.PersistenceManager import PersistenceManagerFactory
from SharedKernel.utils.yamlenv import load_env_yaml
from src.Domain.base_entities import Departments
from src.Features.DeparmentAPI.DepartmentDTO import DepartmentSearchRequest
from src.SharedKernel.base.Logger import get_logger
from src.SharedKernel.base.Page import Page
from src.SharedKernel.persistence.CrudRepository import CrudRepository
from src.SharedKernel.persistence.PersistenceManager import get_db_session
from src.SharedKernel.persistence.QueryExtension import QueryExtension

logger = get_logger(__name__)

class DepartmentRepository(CrudRepository[Departments, uuid.UUID]):
    def __init__(self, session: AsyncSession = Depends(get_db_session)):
        super().__init__(Departments, session)

    async def search_departments(self, req: DepartmentSearchRequest):
        base_query = """
        FROM Departments d
        WHERE 1=1
        AND d.delete_at IS NULL
        """

        query = (
            QueryExtension(base_query)
            .filter(
                req.q, 
                "d.name LIKE :q", 
                q=f"%{req.q}%"
            )
            .paginate(req.page, req.page_size)
        )

        exec_query, params = query.build_select("*")
        count_query, count_params = query.build_count()

        exec_result = await self.fetch_all(exec_query, params)
        count_result = await self.fetch_all(count_query, count_params)

        return Page(
            content=exec_result,
            page_number=req.page,
            page_size=req.page_size,
            total_elements=count_result[0]['total']
        )

    async def search_department(self, req: DepartmentSearchRequest):
        # Legacy method - delegate to search_departments
        return await self.search_departments(req)