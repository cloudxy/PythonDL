from pydantic import BaseModel, Field, ConfigDict
from typing import Optional
from datetime import datetime


class SystemLogBase(BaseModel):
    log_level: str = Field(..., min_length=1, max_length=20)
    module: Optional[str] = Field(None, max_length=100)
    action: Optional[str] = Field(None, max_length=200)
    user_id: Optional[int] = None
    username: Optional[str] = Field(None, max_length=50)
    request_method: Optional[str] = Field(None, max_length=10)
    request_url: Optional[str] = Field(None, max_length=500)
    request_params: Optional[str] = Field(None, max_length=2000)
    response_status: Optional[int] = None
    error_message: Optional[str] = Field(None, max_length=2000)
    ip_address: Optional[str] = Field(None, max_length=50)
    user_agent: Optional[str] = Field(None, max_length=500)


class SystemLogCreate(SystemLogBase):
    pass


class SystemLogResponse(SystemLogBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
