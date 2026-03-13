from pydantic import BaseModel, Field, ConfigDict
from typing import Optional
from datetime import datetime


class SystemConfigBase(BaseModel):
    config_key: str = Field(..., min_length=1, max_length=100)
    config_value: str = Field(..., max_length=2000)
    config_type: str = Field(..., min_length=1, max_length=50)
    description: Optional[str] = Field(None, max_length=500)
    is_active: bool = True


class SystemConfigCreate(SystemConfigBase):
    pass


class SystemConfigUpdate(BaseModel):
    config_key: Optional[str] = Field(None, min_length=1, max_length=100)
    config_value: Optional[str] = Field(None, max_length=2000)
    config_type: Optional[str] = Field(None, min_length=1, max_length=50)
    description: Optional[str] = Field(None, max_length=500)
    is_active: Optional[bool] = None


class SystemConfigResponse(SystemConfigBase):
    model_config = ConfigDict(from_attributes=True)
    id: int
    created_at: datetime
    updated_at: datetime
