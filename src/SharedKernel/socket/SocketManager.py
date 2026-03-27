import asyncio
import json
from typing import Dict, List
from fastapi import WebSocket
from SharedKernel.base.Logger import get_logger

logger = get_logger(__name__)

class SocketManager:
    _instance = None
    _initialized = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self.__class__._initialized:
            return

        self.active_connections: Dict[str, List[WebSocket]] = {}
        self.usernames: Dict[WebSocket, str] = {}
        self.lock = asyncio.Lock()

        self.__class__._initialized = True

    async def connect(self, websocket: WebSocket, room_id: str, username: str):
        await websocket.accept()

        async with self.lock:
            self.active_connections.setdefault(room_id, []).append(websocket)
            self.usernames[websocket] = username

        logger.info(f"{username} connected to room {room_id}")

        # json_res = json.dumps({
        #         "message": f"{username} đã tham gia phòng."
        #     })
        # await self.broadcast_system(json_res, room_id, exclude=websocket)

    async def disconnect(self, websocket: WebSocket, room_id: str):
        async with self.lock:
            username = self.usernames.get(websocket, "Unknown")

            if room_id in self.active_connections:
                if websocket in self.active_connections[room_id]:
                    self.active_connections[room_id].remove(websocket)

                if not self.active_connections[room_id]:
                    del self.active_connections[room_id]

            self.usernames.pop(websocket, None)

        logger.info(f"{username} disconnected")

        # await self.broadcast_system(f"{username} đã rời phòng.", room_id)

    async def send_personal_message(self, websocket: WebSocket, message: str):
        try:
            await websocket.send_text(message)
        except Exception:
            logger.error("Send personal message failed")

    async def broadcast(self, sender_ws: WebSocket, message: str, room_id: str):
        async with self.lock:
            connections = self.active_connections.get(room_id, []).copy()

        dead_connections = []

        for connection in connections:
            if connection == sender_ws:
                continue

            try:
                await connection.send_text(message)
            except Exception:
                dead_connections.append(connection)

        for dead in dead_connections:
            await self.disconnect(dead, room_id)

    async def broadcast_system(self, message: str, room_id: str, exclude=None):
        async with self.lock:
            connections = self.active_connections.get(room_id, []).copy()

        for connection in connections:
            if connection == exclude:
                continue

            try:
                await connection.send_text(message)
            except Exception:
                pass

    async def list_users(self, websocket: WebSocket, room_id: str):
        async with self.lock:
            users = [
                self.usernames[ws]
                for ws in self.active_connections.get(room_id, [])
                if ws in self.usernames
            ]

        if not users:
            await self.send_personal_message(websocket, "Không có ai online.")
        else:
            await self.send_personal_message(websocket, "Online: " + ", ".join(users))

    async def send_private(
        self,
        sender_ws: WebSocket,
        room_id: str,
        target_name: str,
        message: str
    ):
        sender_name = self.usernames.get(sender_ws, "Unknown")

        async with self.lock:
            target_ws = None
            for ws in self.active_connections.get(room_id, []):
                if self.usernames.get(ws) == target_name:
                    target_ws = ws
                    break

        if not target_ws:
            await self.send_personal_message(sender_ws, f"Không tìm thấy user: {target_name}")
            return

        try:
            await target_ws.send_text(f"[Riêng] {sender_name}: {message}")
            await sender_ws.send_text(f"[Bạn -> {target_name}] {message}")
        except Exception:
            await self.send_personal_message(sender_ws, f"Không gửi được tới {target_name}")