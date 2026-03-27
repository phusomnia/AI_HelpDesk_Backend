from typing import Any, TypeVar, Generic, Type, Optional, Sequence
from pydantic import BaseModel
from sqlmodel import select, text
from sqlmodel.ext.asyncio.session import AsyncSession
from SharedKernel.persistence.BaseRepository import BaseRepository
from SharedKernel.persistence.PersistenceManager import PersistenceManagerFactory
from SharedKernel.utils.yamlenv import load_env_yaml

config = load_env_yaml()

T = TypeVar("T", bound=BaseModel)
ID = TypeVar("ID")

class RepositoryMeta(Generic[T, ID], type(BaseRepository)):
    pass

class CrudRepository(BaseRepository, Generic[T, ID], metaclass=RepositoryMeta):
    def __init__(self, model: Type[T], session: AsyncSession):
        super().__init__(session)
        self.model = model

    async def find_all(self) -> Sequence[T]:
        result = await self.session.exec(select(self.model))
        return result.all()

    async def save(self, entity: T) -> T:
        self.session.add(entity)
        await self.session.commit()
        await self.session.refresh(entity)
        return entity

    async def find_by_id(self, id: ID) -> Optional[T]:
        return await self.session.get(self.model, id)

    async def update(self, entity: T) -> T:
        self.session.add(entity)
        await self.session.commit()
        return entity

    async def delete(self, entity: BaseModel) -> None:
        await self.session.delete(entity)
        await self.session.commit()


    