"""
MemoryRepo - Repository pattern for conversation history database operations
KISS Principle: Simple, focused, single responsibility
"""

import asyncio
from datetime import datetime, timezone
from typing import Dict, List
from pathlib import Path
from sqlalchemy.ext.asyncio import create_async_engine, AsyncEngine
from sqlmodel import SQLModel, text
from sqlalchemy import text as sa_text
import uuid6
from src.Domain.history_entities import ConversationHistory
from src.SharedKernel.persistence.QueryExtension import QueryExtension
from src.SharedKernel.base.Page import Page


class MemoryRepository:
    """
    Repository for conversation history persistence.
    Handles all database operations for chat memory.
    """

    def __init__(self, db_path: str = "specs/data/chat_history.db"):
        self.db_path = Path(db_path).resolve()
        self._ensure_dir()
        self._sqlite_engine: AsyncEngine = None
        self._initialized = False
        self._init_lock = asyncio.Lock()

    @property
    def sqlite_engine(self) -> AsyncEngine:
        """Lazy initialization of database engine"""
        if self._sqlite_engine is None:
            self._sqlite_engine = create_async_engine(
                f"sqlite+aiosqlite:///{self.db_path}"
            )
        return self._sqlite_engine

    async def _ensure_initialized(self) -> None:
        """Ensure database is initialized before operations - thread-safe"""
        if self._initialized:
            return

        async with self._init_lock:
            if self._initialized:
                return

            async with self.sqlite_engine.begin() as conn:
                await conn.run_sync(
                    SQLModel.metadata.create_all,
                    tables=[ConversationHistory.__table__]
                )

            self._initialized = True

    def _ensure_dir(self):
        """Ensure data directory exists"""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    # async def init_db(self) -> None:
    #     """Initialize database - create table if not exists"""
    #     async with self.sqlite_engine.begin() as conn:
    #         result = await conn.execute(sa_text("""
    #             SELECT 1 FROM sqlite_master 
    #             WHERE type='table' AND name='conversation_history'
    #         """))
    #         exists = result.scalar() is not None

    #         if not exists:
    #             await conn.run_sync(
    #                 SQLModel.metadata.create_all,
    #                 tables=[ConversationHistory.__table__]
    #             )

    async def add_message(self, session_id: str, role: str, content: str) -> None:
        """Insert a single message to database"""
        await self._ensure_initialized()
        query = sa_text("""
            INSERT INTO conversation_history 
            (id, session_id, role, content, timestamp)
            VALUES (:id, :session_id, :role, :content, :timestamp)
        """)
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        async with self.sqlite_engine.begin() as conn:
            await conn.execute(query, {
                "id": str(uuid6.uuid7()),
                "session_id": session_id,
                "role": role,
                "content": content,
                "timestamp": now
            })

    async def get_history_paginated(
        self,
        session_id: str,
        page_number: int = 1,
        page_size: int = 5
    ) -> Page:
        """Get paginated conversation history"""
        await self._ensure_initialized()
        if page_number < 1:
            page_number = 1
        if page_size <= 0:
            page_size = 5

        base_query = """
        FROM conversation_history ch
        WHERE 1=1
        """

        query = (
            QueryExtension(base_query)
            .filter(
                session_id,
                "ch.session_id = :session_id",
                session_id=session_id
            )
            .order_by("ch.timestamp DESC")
            .paginate(page_number, page_size)
        )

        data_query, params = query.build_select("ch.*")
        count_query, count_params = query.build_count()

        async with self.sqlite_engine.connect() as conn:
            data_result = await conn.execute(sa_text(data_query), params)
            data = [dict(row) for row in data_result.mappings().all()]
            data = data[::-1]  # Reverse to chronological order

            count_result = await conn.execute(sa_text(count_query), count_params)
            total_elements = count_result.scalar()

            return Page(
                content=data,
                page_number=page_number,
                page_size=page_size,
                total_elements=total_elements
            )

    async def get_history_all(self, session_id: str) -> List[Dict]:
        """Get all conversation history for a session"""
        await self._ensure_initialized()
        query = sa_text("""
            SELECT ch.* 
            FROM conversation_history ch
            WHERE ch.session_id = :session_id
            ORDER BY ch.timestamp ASC
        """)

        async with self.sqlite_engine.connect() as conn:
            result = await conn.execute(query, {"session_id": session_id})
            return [dict(row) for row in result.mappings().all()]

    async def get_recent_messages(
        self,
        session_id: str,
        limit: int = 10
    ) -> List[Dict]:
        """Get recent messages for context window"""
        await self._ensure_initialized()
        query = sa_text("""
            SELECT ch.* 
            FROM conversation_history ch
            WHERE ch.session_id = :session_id
            ORDER BY ch.timestamp DESC
            LIMIT :limit
        """)

        async with self.sqlite_engine.connect() as conn:
            result = await conn.execute(
                query,
                {"session_id": session_id, "limit": limit}
            )
            data = [dict(row) for row in result.mappings().all()]
            return data[::-1]  # Reverse to chronological order

    async def delete_session_history(self, session_id: str) -> int:
        """Delete all history for a session. Returns deleted count"""
        await self._ensure_initialized()
        query = sa_text("""
            DELETE FROM conversation_history
            WHERE session_id = :session_id
        """)

        async with self.sqlite_engine.begin() as conn:
            result = await conn.execute(query, {"session_id": session_id})
            return result.rowcount

    async def close(self) -> None:
        """Close database connection"""
        await self.sqlite_engine.dispose()
