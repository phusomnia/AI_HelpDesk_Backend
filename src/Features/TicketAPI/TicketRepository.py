from typing import Optional
import uuid
from fastapi import Depends
from sqlalchemy.ext.asyncio import AsyncSession
from SharedKernel.base.Logger import get_logger
from SharedKernel.persistence.CrudRepository import CrudRepository
from SharedKernel.persistence.QueryExtension import QueryExtension
from src.Features.TicketAPI.TicketDTO import TicketSearchRequest
from src.SharedKernel.base.Page import Page
from src.Domain.base_entities import Tickets
from src.SharedKernel.persistence.PersistenceManager import get_db_session

logger = get_logger(__name__)

class TicketRepository(CrudRepository[Tickets, uuid.UUID]):
    def __init__(self, session: AsyncSession = Depends(get_db_session)):
        super().__init__(Tickets, session)

    async def search_tickets(self, req: TicketSearchRequest):
        base_query = """
        FROM Tickets t
        LEFT JOIN Departments d ON t.dept_id = d.id
        WHERE 1=1
        AND t.delete_at IS NULL
        """

        query = (
            QueryExtension(base_query)
            .filter(
                req.category, 
                "t.category LIKE :category", 
                category=f"%{req.category}%"
            )
            .filter(
                req.department_name, 
                "d.name LIKE :department_name", 
                department_name=f"%{req.department_name}%"
            )
            .filter(
                req.status, 
                "t.status = :status", 
                status=req.status
            )
            .filter(
                req.priority, "t.priority = :priority", priority=req.priority)
            .paginate(
                req.page, 
                req.page_size
            )
        )

        exec_query, params = query.build_select("t.*, d.name as department_name")
        count_query, count_params = query.build_count()

        exec_result = await self.fetch_all(exec_query, params)
        count_result = await self.fetch_all(count_query, count_params)

        return Page(
            content=exec_result,
            page_number=req.page,
            page_size=req.page_size,
            total_elements=count_result[0]['total']
        )