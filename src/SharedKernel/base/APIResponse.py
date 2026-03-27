from typing import TypeVar, Generic, Optional
from pydantic import BaseModel, ConfigDict

T = TypeVar('T')

class APIResponse(BaseModel, Generic[T]):
    model_config = ConfigDict(
        exclude_none=True, 
        arbitrary_types_allowed=True  
    )

    message: str
    data: Optional[T] = None
    status_code: int