from typing import List, Optional
from pydantic import BaseModel

class MessageRequest(BaseModel):
    user_id: str
    content: str
    receiver_id: Optional[str] = None 
    attachments: Optional[List[str]] = None

# class SendMessageRequest(BaseModel):
#     conversation_key: str
#     user_id: str
#     message: str
#     attachments: Optional[List[str]] = None