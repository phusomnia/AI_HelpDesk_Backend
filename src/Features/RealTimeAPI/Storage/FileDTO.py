from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field

class TypeStorage(str, Enum):
    SYSTEM          = "system_storage"
    USER            = "user_storage"

class FileSearchRequest(BaseModel):
    page: int = Field(1, ge=1, description="Page number")
    page_size: int = Field(10, ge=1, le=100, description="Page size")
    # category: Optional[str] = Field(None, description="Filter by category")
    # department_name: Optional[str] = Field(None, description="Filter by department")
    # status: Optional[str] = Field(None, description="Filter by status")
    # priority: Optional[str] = Field(None, description="Filter by priority")