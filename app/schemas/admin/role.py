from pydantic import BaseModel, Field, ConfigDict
from typing import Optional
from datetime import datetime


class RoleBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=50)
    code: str = Field(..., min_length=1, max_length=50)
    description: Optional[str] = Field(None, max_length=200)
    is_active: bool = True


class RoleCreate(RoleBase):
    pass


class RoleUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=50)
    code: Optional[str] = Field(None, min_length=1, max_length=50)
    description: Optional[str] = Field(None, max_length=200)
    is_active: Optional[bool] = None


class RoleResponse(RoleBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
    updated_at: datetime
