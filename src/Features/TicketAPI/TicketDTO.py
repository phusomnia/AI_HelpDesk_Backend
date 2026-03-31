from typing import Any, Optional, Union
from pydantic import BaseModel, ConfigDict, Field
from sqlalchemy import Null
from sqlmodel import JSON
from datetime import datetime


class TicketBaseDTO(BaseModel):
    # Core fields
    id: Optional[str] = Field(default=None, example="uuid-string")
    subject: Optional[str] = Field(default=None, example="Vấn đề đăng nhập")
    description: Optional[str] = Field(default=None, example="Không thể đăng nhập vào hệ thống")
    status: Optional[str] = Field(default=None, example="OPEN")
    priority: Optional[str] = Field(default=None, example="MEDIUM")
    category: Optional[str] = Field(default=None, example="Technical")
    
    # Relationship fields
    customer_id: Optional[str] = Field(default=None, example="customer-uuid")
    assigned_agent_id: Optional[str] = Field(default=None, example="agent-uuid")
    dept_id: Optional[str] = Field(default=None, example="dept-uuid")
    
    # Additional fields
    satisfaction_rating: Optional[int] = Field(default=None, example=5)
    customer_feedback: Optional[str] = Field(default=None, example="Rất hài lòng")
    attachment_url: Optional[dict] = Field(default=None, example={"files": ["url1", "url2"]})
    due_date: Optional[datetime] = Field(default=None, example="2024-01-01T00:00:00")
    
    # Timestamps
    created_at: Optional[datetime] = Field(default=None, example="2024-01-01T00:00:00")
    delete_at: Optional[datetime] = Field(default=None, example=None)
    
class TicketSearchRequest(BaseModel):
    page: int = Field(1, ge=1, description="Page number")
    page_size: int = Field(5, ge=1, le=100, description="Page size")
    category: Optional[str] = Field(None, description="Filter by category")
    department_name: Optional[str] = Field(None, description="Filter by department")
    status: Optional[str] = Field(None, description="Filter by status")
    priority: Optional[str] = Field(None, description="Filter by priority")

