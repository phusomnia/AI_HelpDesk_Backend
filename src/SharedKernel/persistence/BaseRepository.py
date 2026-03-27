from ctypes import Union
from typing import Any, Dict, List, Union, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlmodel import text

RawQueryResult = Union[List[Dict[str, Any]], Dict[str, int]]

class BaseRepository:
    def __init__(self, session: AsyncSession):
        self.session = session

    async def fetch_all(
        self, 
        sql: str, 
        params: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        try:
            stmt = text(sql)

            if params is not None: 
                stmt = stmt.bindparams(**params)
            
            result = await self.session.execute(stmt)
            
            return [dict(row) for row in result.mappings()]
        except Exception as e:
            await self.session.rollback()
            raise e
    
    async def fetch_one(
        self, 
        sql: str, 
        params: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        try:
            stmt = text(sql)

            if params is not None: 
                stmt = stmt.bindparams(**params)
            
            result = await self.session.exec(stmt)
            
            first_result = result.mappings().first()
            return dict(first_result) if first_result else None
        except Exception as e:
            await self.session.rollback()
            raise e

    # async def execute_raw(
    #     self,
    #     sql: str
    # ) ->  RawQueryResult:
    #     try:
    #         stmt = text(sql)
            
    #         result = await self.session.exec(stmt)
    #         await self.session.commit()
            
    #         return {"affected_rows": result.rowcount}
                
    #     except Exception as e:
    #         await self.session.rollback()
    #         raise e

    async def execute(
        self,
        sql: str,
        params: Optional[Dict[str, Any]] = None
    ) -> RawQueryResult:
        try:
            stmt = text(sql)
            
            if params is not None: 
                stmt = stmt.bindparams(**params)
            
            result = await self.session.execute(stmt)
            await self.session.commit()
            
            return {"affected_rows": result.rowcount}
                
        except Exception as e:
            await self.session.rollback()
            raise e

    def update_model_from_dto(self, model, dto):
        update_data = dto.model_dump(exclude_unset=True)
        for field, value in update_data.items():
            if value not in [None, "None", ""]:
                setattr(model, field, value)
        return model