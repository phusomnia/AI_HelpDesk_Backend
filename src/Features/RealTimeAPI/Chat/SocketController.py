from fastapi import APIRouter, Depends, FastAPI, WebSocket, status
from src.Features.RealTimeAPI.Chat.ChatService import ChatService
from src.SharedKernel.base.APIResponse import APIResponse
from src.SharedKernel.persistence.Decorators import Controller

@Controller
class SocketController:
    def __init__(self, app: FastAPI) -> None:
        self.app = app
        self.ws_router = APIRouter(
            prefix="/api/v1/ws",
            tags=["Websocket"]
        )
        self.chat_router = APIRouter(
            prefix="/api/v1/chatroom",
            tags=["Chatroom"]
        )
        self.register_websocket()
        self.register_chatroom()
        self.app.include_router(self.ws_router)
        self.app.include_router(self.chat_router)

    def register_websocket(self):
    #     @self.ws_router.websocket("/{user_id}")
    #     async def websocket_chat_new_conversation(
    #         websocket: WebSocket,
    #         user_id: str,
    #         chat_service: ChatService = Depends()
    #     ):
    #         print(f"Starting new conversation for user: {user_id}")
    #         await chat_service.websocket_chat(websocket, user_id, "None")

        @self.ws_router.websocket("/{user_id}/{conversation_key}")
        async def websocket_chat_with_conversation(
            websocket: WebSocket,
            user_id: str,
            conversation_key: str,
            chat_service: ChatService = Depends()
        ):
            print(f"Joining conversation: {conversation_key}")
            await chat_service.websocket_chat(websocket, user_id, conversation_key)

    def register_chatroom(self):
        @self.chat_router.get("/messages/{conversation_key}")
        async def get_messages(
            conversation_key: str,
            service: ChatService = Depends()
        ):
            result = await service.get_messages_by_conversation_key(conversation_key)
            return APIResponse(
                message="Messages retrieved successfully",
                status_code=status.HTTP_200_OK,
                data=result
            )

        @self.chat_router.get("/conversation_key/{user_id}")
        async def gen_conversation_key_by_user_id(
            user_id: str,
            service: ChatService = Depends()
        ):
            result = await service.gen_conversation_key(user_id) 
            return APIResponse(
                message="Conversation retrieved successfully",
                status_code=status.HTTP_200_OK,
                data=result
            )

        @self.chat_router.get("/conversation_key/agent/{user_id}")
        async def get_conversation_key_by_agent(
            user_id: str,
            service: ChatService = Depends()
        ):
            result = await service.get_conversation_key_by_user_id(user_id) 
            return APIResponse(
                message="Conversation retrieved successfully",
                status_code=status.HTTP_200_OK,
                data=result
            )

        # @self.chat_router.post("/")
        # async def send_message(
        #     req: MessageRequest,
        #     service: ChatService = Depends()
        # ):
        #     result = await service.send_message(req)
            
        #     return APIResponse(
        #         message="Message sent successfully",
        #         status_code=status.HTTP_201_CREATED,
        #         data=result
        #     )




