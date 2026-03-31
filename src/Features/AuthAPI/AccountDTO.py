from typing import Optional
import bcrypt
from pydantic import BaseModel, Field
from datetime import datetime
from src.Domain.base_entities import Accounts, AccountsRole

class AccountBaseDTO(BaseModel):
    # Core fields
    id: Optional[str] = Field(default=None, example="uuid-string")
    username: Optional[str] = Field(default=None, example="john_doe")
    email: Optional[str] = Field(default=None, example="john@example.com")
    full_name: Optional[str] = Field(default=None, example="John Doe")
    password: Optional[str] = Field(default=None, example="hashed_password")
    role: Optional[AccountsRole] = Field(default=None, example="CUSTOMER")
    department_id: Optional[str] = Field(default=None, example="dept-uuid")
 
    # Timestamps
    created_at: Optional[datetime] = Field(default=None, example="2024-01-01T00:00:00")
    delete_at: Optional[datetime] = Field(default=None, example=None)

class CreateAccountRequest(BaseModel):
    username: str
    password: str
    email: str
    role: Optional[AccountsRole] | None 
    department_id: Optional[str] | None 

    def to_entity(self):
        data = self.model_dump()
        data["password"] = bcrypt.hashpw(data["password"].encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        return Accounts(**data)
    ...

class LoginAccountRequest(BaseModel):
    username: str
    password: str
    ...

class UpdateAccountRequest(BaseModel):
    username: Optional[str] = None
    password: Optional[str] = None
    email: Optional[str] = None
    role: Optional[AccountsRole] 
    department_id: Optional[str]

    def to_entity(self, account: Accounts):
        update_data = self.model_dump(exclude_unset=True)  

        for field, value in update_data.items():
            if field == "password":
                value = bcrypt.hashpw(value.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

            setattr(account, field, value)

        return account

    ...


class SearchAccountRequest(BaseModel):
    page: int = 1
    page_size: int = 10
    role: Optional[AccountsRole] = Field(None, description="Filter by role")
    department_name: Optional[str] = Field(None, description="Filter by department")