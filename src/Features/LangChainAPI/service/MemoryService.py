import json
from pathlib import Path
from Features.LangChainAPI.LangChainDTO import ChatMessageRequest
from SharedKernel.persistence.Decorators import Service
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_community.chat_message_histories import SQLChatMessageHistory
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from sqlalchemy import create_engine

memory_store: dict[str, InMemoryChatMessageHistory] = {}

# @Service
class MemoryService:
    def __init__(self, provider, callbacks) -> None:
        self.provider = provider
        self.callbacks = callbacks or {}
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "Bạn là một trợ lý ảo hữu ích"),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}")
        ])
        self.url = self.create_file(".data", "memory.db")
        self.chat_chain = self.prompt | self.provider
    pass

    async def short_chat(self, req: ChatMessageRequest):
        history = self.get_inmem_history(req.session_id)
        history.add_user_message(req.message)
        past_messages = history.messages[:-1]

        response_message = await self.chat_chain.ainvoke({
            "history": past_messages,
            "input": req.message
        })

        history.add_ai_message(response_message)

        return {
            "response": response_message.content,
            "session_id": req.session_id
        }
        ...

    async def long_chat(self, req: ChatMessageRequest):
        print(f"URL {self.url}")
        chat_history = self.get_sql_history(req.session_id)

        chat_history.add_user_message(req.message)

        response = await self.chat_chain.ainvoke({
            "history": chat_history.messages,
            "input": req.message
        })

        chat_history.add_ai_message(response)

        return {
            "response": response.content,
            "session_id": req.session_id
        }

        ...

    def get_sql_history(self, session_id: str):
        engine = create_engine(f"sqlite:///{self.url}")
        return SQLChatMessageHistory(
            session_id=session_id,
            connection=engine
        )

    def get_inmem_history(self, session_id: str):
        if session_id not in memory_store:
            memory_store[session_id] = InMemoryChatMessageHistory()
        return memory_store[session_id]

    def create_file(self, path: str, name: str) -> str:
        full_path = Path(path).resolve() / name
        if not full_path.exists():
            try:
                full_path.touch()
                print(f"File created: {full_path}")
            except OSError as e:
                raise ValueError(f"Cannot create file: {e}")
        return str(full_path)
    
