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
                "t.dept_id = :department_name", 
                department_name=req.department_name
            )
            .filter(
                req.status, 
                "t.status = :status", 
                status=req.status
            )
            .filter(
                req.priority, "t.priority = :priority", priority=req.priority)
            .filter(
                req.customer_id,
                "t.customer_id = :customer_id",
                customer_id=req.customer_id
            )
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

    async def get_status_statistics(self):
        query = """
        SELECT 
            status,
            COUNT(*) as count,
            ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM Tickets WHERE delete_at IS NULL), 2) as percentage
        FROM Tickets 
        WHERE delete_at IS NULL
        GROUP BY status
        ORDER BY count DESC
        """
        return await self.fetch_all(query)

    async def get_priority_statistics(self):
        query = """
        SELECT 
            priority,
            COUNT(*) as count,
            ROUND(COUNT(*) * 100.0 / (SELECT COUNT(*) FROM Tickets WHERE delete_at IS NULL), 2) as percentage
        FROM Tickets 
        WHERE delete_at IS NULL
        GROUP BY priority
        ORDER BY 
            CASE priority
                WHEN 'URGENT' THEN 1
                WHEN 'HIGH' THEN 2
                WHEN 'MEDIUM' THEN 3
                WHEN 'LOW' THEN 4
            END
        """
        return await self.fetch_all(query)

    async def get_time_statistics(self, year: int = None, month: int = None):
        base_filter = "WHERE delete_at IS NULL"
        params = {}
        
        if year:
            base_filter += " AND YEAR(created_at) = :year"
            params['year'] = year
        
        if month:
            base_filter += " AND MONTH(created_at) = :month"
            params['month'] = month
        
        query = f"""
        SELECT 
            DATE_FORMAT(created_at, '%Y-%m') as month,
            COUNT(*) as total_tickets,
            SUM(CASE WHEN status = 'RESOLVED' THEN 1 ELSE 0 END) as resolved_tickets,
            SUM(CASE WHEN status = 'CLOSED' THEN 1 ELSE 0 END) as closed_tickets,
            SUM(CASE WHEN status IN ('RESOLVED', 'CLOSED') THEN 1 ELSE 0 END) as processed_tickets,
            ROUND(
                SUM(CASE WHEN status IN ('RESOLVED', 'CLOSED') THEN 1 ELSE 0 END) * 100.0 / COUNT(*), 
                2
            ) as processed_percentage
        FROM Tickets 
        {base_filter}
        GROUP BY DATE_FORMAT(created_at, '%Y-%m')
        ORDER BY month DESC
        """

        return await self.fetch_all(query, params)