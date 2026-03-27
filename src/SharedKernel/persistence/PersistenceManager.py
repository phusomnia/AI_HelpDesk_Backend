from abc import ABC, abstractmethod
from typing import Any, AsyncGenerator, Dict, Type
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlmodel.ext.asyncio.session import AsyncSession
from SharedKernel.utils.yamlenv import load_env_yaml

config = load_env_yaml()

class IPersistenceManager(ABC):
    @abstractmethod
    def get_async_session(self) -> AsyncSession:
        pass
    pass

    @abstractmethod
    def get_engine(self) -> AsyncEngine:
        pass
    pass

class MYSQLManager(IPersistenceManager):
    def __init__(self) -> None:
        self.DATABASE_URL = config.database.mysql.url
        
        self.engine = create_async_engine(
            self.DATABASE_URL, 
            echo=False, 
            pool_pre_ping=True
        )
        
        self.async_session = async_sessionmaker(
            bind=self.engine, 
            class_=AsyncSession, 
            autoflush=False,
            autocommit=False,
            expire_on_commit=False
        )
        
        ...

    def get_engine(self) -> AsyncEngine:
        return self.engine
    
    def get_async_session(self) -> AsyncSession:
        return self.async_session()

    pass

# class PostgreManager(IPersistenceManager):
#     def __init__(self) -> None:
#         self.DATABASE_URL = config.database.mysql.url
#         print("Postgre ...")
#         self.engine = create_async_engine(
#             self.DATABASE_URL, 
#             echo=True, 
#             pool_pre_ping=True
#         )
        
#         self.async_session = async_sessionmaker(
#             bind=self.engine, 
#             class_=AsyncSession, 
#             expire_on_commit=False
#         )

#         ...
    
#     def get_engine(self) -> AsyncEngine:
#         return self.engine

#     def get_async_session(self) -> AsyncSession:
#         return self.async_session()

#     pass

class PersistenceManagerFactory():
    _registry: Dict[str, Type[IPersistenceManager]] = {}

    @classmethod
    def register(cls, type_name: str, persistence_class: Type[IPersistenceManager]):
        cls._registry[type_name] = persistence_class

    @classmethod
    def create(cls, type_name: str) -> IPersistenceManager:
        persistence_class = cls._registry.get(type_name)
        
        if not persistence_class:
            raise ValueError(f"Database '{type_name}' is not registered.")
            
        return persistence_class()

PersistenceManagerFactory.register("MYSQL", MYSQLManager)
# PersistenceManagerFactory.register("POSTGRE", PostgreManager)

async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """Dependency để get database session"""
    pm = PersistenceManagerFactory.create(config.database.type)
    async with pm.get_async_session() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise