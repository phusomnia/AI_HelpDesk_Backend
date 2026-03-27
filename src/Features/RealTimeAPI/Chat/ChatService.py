import base64
import json
import os
from typing import List, Optional

import uuid6
from src.Domain.base_entities import Attachment, Messages
from src.Features.RealTimeAPI.Chat.ChatDTO import MessageRequest
from src.Features.RealTimeAPI.Chat.ChatRepository import ChatRepository
from fastapi import Depends, WebSocket, WebSocketDisconnect, status
from src.Features.RealTimeAPI.Storage.StorageService import StorageService
from src.Features.RealTimeAPI.Storage.FileDTO import TypeStorage
from src.SharedKernel.Utils import Utils
from src.SharedKernel.exception.APIException import APIException
from src.SharedKernel.socket.SocketManager import SocketManager

def get_socket_manager() -> SocketManager:
    return SocketManager()

class ChatService:
    def __init__(self, 
        repo: ChatRepository = Depends(),
        storage_service: StorageService = Depends(),
        socket_manager: SocketManager = Depends(get_socket_manager)
    ):
        self.repo = repo
        self.storage_service = storage_service
        self.socket_manager = socket_manager
    ...

    #
    # SOCKET
    #
    async def websocket_chat(self, 
        websocket: WebSocket, 
        user_id: str, 
        conversation_key: Optional[str] = None, 
    ):
        customer_care_agent_id = None

        is_agent = await self.repo.fetch_one(
            """
            SELECT a.* 
            FROM Accounts a
            WHERE a.id = :user_id
            AND a.role = 'AGENT'
            """,
            {"user_id": user_id}
        )

        if not is_agent:
            customer_care_agent = await self.repo.fetch_one(
                """
                SELECT a.* 
                FROM Accounts a
                JOIN Departments d ON a.department_id = d.id
                AND a.role = 'AGENT'
                AND d.name = :dept_name
                """,
                {"dept_name": "Chăm sóc khách hàng"}
            )

            if customer_care_agent is None:
                return APIException(
                    message="Ko tìm thấy nhân viên cskh",
                    status=status.HTTP_404_NOT_FOUND
                )

            customer_care_agent_id = customer_care_agent['id']

        if conversation_key == "None" and customer_care_agent_id is not None:
            conversation_key = Utils.generate_conversation_key(user_id, customer_care_agent_id)
            print(f"Generated conversation key: {conversation_key}")

        # await websocket.accept() 
        try:
            await self.socket_manager.connect(websocket, conversation_key, user_id)

            while True:
                raw_message = await websocket.receive_text()
                print(raw_message)

                ws_data = json.loads(raw_message)
                print(f"Received message: {ws_data}")
                if ws_data.get('type') == "message" or ws_data.get('type') == "file":
                    chat = None
                    if is_agent:
                        user_id_in_conversation = Utils.extract_customer_id_from_conversation_key(conversation_key, customer_care_agent_id)
                        chat = Messages(
                            conversation_key=conversation_key,
                            sender_id=ws_data.get('sender_id'), 
                            content=ws_data.get('content'),
                            receiver_id=user_id_in_conversation
                        )
                    else:
                        chat = Messages(
                            conversation_key=conversation_key,
                            sender_id=ws_data.get('sender_id'), 
                            content=ws_data.get('content'),
                            receiver_id=customer_care_agent_id
                        )
                    
                    if chat:
                        await self.repo.save(chat)
                        print("Save user chat")  

                    response = { "sender_id": ws_data.get('sender_id'), "content": ws_data.get('content') }
                    json_res = json.dumps(response)

                    await self.socket_manager.broadcast(websocket, json_res, conversation_key) 
                    
                if ws_data.get('type') == "typing":
                    data = json.dumps(ws_data)
                    await self.socket_manager.broadcast(websocket, data, conversation_key) 
        except WebSocketDisconnect:
            await self.socket_manager.disconnect(websocket, conversation_key)
            response = { "content": f"User {user_id} left the chat" }
            json_res = json.dumps(response)
        ...



    async def send_message(self, req: MessageRequest):
        message = Messages(
            sender_id=req.user_id, 
            content=req.content,
            receiver_id=req.receiver_id
        ) 
        return await self.save(message)
    
    async def get_messages_by_conversation_key(self, conversation_key: str):
        affected_rows = await self.repo.fetch_all(
            "SELECT * FROM Messages m WHERE m.conversation_key = :key AND m.delete_at IS NULL",
            {"key": conversation_key}
        )
        return affected_rows
    
    async def get_conversation_key_by_user_id(self, user_id: str):
        return await self.repo.fetch_all(
            """
            SELECT DISTINCT conversation_key
            FROM Messages
            WHERE conversation_key LIKE :user_id
            ORDER BY conversation_key;
            """,
            {"user_id": f"%{user_id}%"}
        )

    async def gen_conversation_key(self, user_id: str):
        customer_care_agent = await self.repo.fetch_one(
            """
            SELECT a.* 
            FROM Accounts a
            JOIN Departments d ON a.department_id = d.id
            AND a.role = 'AGENT'
            AND d.name = :dept_name
            """,
            {"dept_name": "Chăm sóc khách hàng"}
        )

        if customer_care_agent is None:
            raise APIException(
                message="Không tìm thấy nhân viên chăm sóc khách hàng",
                status_code=status.HTTP_404_NOT_FOUND
            )
        
        conversation_key = Utils.generate_conversation_key(user_id, customer_care_agent['id'])
        
        return {
            "conversation_key": conversation_key
        }

    async def get_conversation_key_by_agent(self, agent_id: str):
        """Lấy danh sách conversation keys của một agent"""

        return await self.repo.fetch_all(
            """
            SELECT DISTINCT conversation_key
            FROM Messages
            WHERE conversation_key LIKE :agent_id
            ORDER BY conversation_key;
            """,
            {"agent_id": f"%{agent_id}%"}
        )

        