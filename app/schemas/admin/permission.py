# type: ignore
from pydantic import BaseModel, Field
from typing import Optional
from datetime import datetime


class PermissionBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=50)
    code: str = Field(..., min_length=1, max_length=50)
    resource: str = Field(..., min_length=1, max_length=50)
    action: str = Field(..., min_length=1, max_length=20)
    description: Optional[str] = Field(None, max_length=200)


class PermissionCreate(PermissionBase):
    pass


class PermissionUpdate(BaseModel):
    name: Optional[str] = Field(None, min_length=1, max_length=50)
    code: Optional[str] = Field(None, min_length=1, max_length=50)
    resource: Optional[str] = Field(None, min_length=1, max_length=50)
    action: Optional[str] = Field(None, min_length=1, max_length=20)
    description: Optional[str] = Field(None, max_length=200)


class PermissionResponse(PermissionBase):
    model_config = {"from_attributes": True}
    id: int
    created_at: datetime
    updated_at: datetime
