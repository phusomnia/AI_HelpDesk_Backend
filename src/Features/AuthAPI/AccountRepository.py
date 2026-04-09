from typing import Any, Dict, Optional
import uuid
from fastapi import Depends
from sqlmodel.ext.asyncio.session import AsyncSession
from src.Domain.base_entities import Accounts
from src.Features.AuthAPI.AccountDTO import SearchAccountRequest, SearchAccountRequest
from src.SharedKernel.base.Page import Page
from src.SharedKernel.persistence.CrudRepository import CrudRepository
from src.SharedKernel.persistence.PersistenceManager import get_db_session
from SharedKernel.persistence.QueryExtension import QueryExtension

class UserRepository(CrudRepository[Accounts, uuid.UUID]):
    def __init__(self, session: AsyncSession = Depends(get_db_session)):
        super().__init__(Accounts, session)

    async def search_accounts(self, req: SearchAccountRequest):
        base_query = """
        FROM Accounts a
        LEFT JOIN Departments d ON a.department_id = d.id
        WHERE 1=1
        """

        query = (
            QueryExtension(base_query)
            .paginate(req.page, req.page_size)
        )

        exec_query, params = query.build_select("a.*, d.name as department_name")
        count_query, count_params = query.build_count()

        exec_result = await self.fetch_all(exec_query, params)
        count_result = await self.fetch_all(count_query, count_params)

        return Page(
            content=exec_result,
            page_number=req.page,
            page_size=req.page_size,
            total_elements=count_result[0]['total']
        )

    async def find_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        base_query = """
        FROM Accounts a
        WHERE a.email = :email
        """

        params = {
            "email": email
        }
        
        exec_result = await self.fetch_one(base_query, params)
        return exec_result

    async def find_by_username(self, username: str, exclude_id: str = None) -> Optional[Dict[str, Any]]:
        base_query = """
        SELECT * 
        FROM Accounts a
        WHERE a.username = :username 
        """
        
        params = {
            "username": username,
        }
        
        if exclude_id:
            base_query += " AND a.id != :exclude_id"
            params["exclude_id"] = exclude_id
        
        exec_result = await self.fetch_one(base_query, params)
        return exec_result