"""系统管理相关Schema

此模块定义系统管理相关的数据验证模型。
"""
from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List
from datetime import datetime


class UserCreate(BaseModel):
    """用户创建"""
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=6)
    real_name: Optional[str] = Field(None, max_length=50)
    phone: Optional[str] = Field(None, max_length=20)
    role_id: Optional[int] = None


class UserUpdate(BaseModel):
    """用户更新"""
    email: Optional[EmailStr] = None
    real_name: Optional[str] = Field(None, max_length=50)
    phone: Optional[str] = Field(None, max_length=20)
    avatar: Optional[str] = None
    role_id: Optional[int] = None
    is_active: Optional[bool] = None


class UserResponse(BaseModel):
    """用户响应"""
    id: int
    username: str
    email: str
    real_name: Optional[str]
    phone: Optional[str]
    avatar: Optional[str]
    role_id: Optional[int]
    is_active: bool
    last_login: Optional[datetime]
    created_at: datetime
    
    class Config:
        from_attributes = True


class RoleCreate(BaseModel):
    """角色创建"""
    role_name: str = Field(..., max_length=50)
    role_code: str = Field(..., max_length=50)
    description: Optional[str] = None


class RoleUpdate(BaseModel):
    """角色更新"""
    role_name: Optional[str] = Field(None, max_length=50)
    description: Optional[str] = None
    is_active: Optional[bool] = None


class RoleResponse(BaseModel):
    """角色响应"""
    id: int
    role_name: str
    role_code: str
    description: Optional[str]
    is_active: bool
    created_at: datetime
    
    class Config:
        from_attributes = True


class PermissionCreate(BaseModel):
    """权限创建"""
    permission_name: str = Field(..., max_length=100)
    permission_code: str = Field(..., max_length=100)
    resource_type: Optional[str] = Field(None, max_length=50)
    resource_path: Optional[str] = Field(None, max_length=255)
    action: Optional[str] = Field(None, max_length=50)
    description: Optional[str] = None
    parent_id: Optional[int] = None


class PermissionResponse(BaseModel):
    """权限响应"""
    id: int
    permission_name: str
    permission_code: str
    resource_type: Optional[str]
    resource_path: Optional[str]
    action: Optional[str]
    description: Optional[str]
    parent_id: Optional[int]
    is_active: bool
    created_at: datetime
    
    class Config:
        from_attributes = True


class SystemConfigCreate(BaseModel):
    """系统配置创建"""
    config_key: str = Field(..., max_length=100)
    config_value: Optional[str] = None
    config_type: str = Field(default="string", max_length=50)
    category: str = Field(default="system", max_length=50)
    description: Optional[str] = None
    is_public: bool = False


class SystemConfigUpdate(BaseModel):
    """系统配置更新"""
    config_value: Optional[str] = None
    description: Optional[str] = None
    is_public: Optional[bool] = None
    is_active: Optional[bool] = None


class SystemConfigResponse(BaseModel):
    """系统配置响应"""
    id: int
    config_key: str
    config_value: Optional[str]
    config_type: str
    category: str
    description: Optional[str]
    is_public: bool
    is_active: bool
    created_at: datetime
    
    class Config:
        from_attributes = True


class OperationLogResponse(BaseModel):
    """操作日志响应"""
    id: int
    user_id: Optional[int]
    username: Optional[str]
    operation_type: Optional[str]
    operation_module: Optional[str]
    operation_desc: Optional[str]
    request_method: Optional[str]
    request_url: Optional[str]
    response_code: Optional[int]
    ip_address: Optional[str]
    execution_time: Optional[int]
    status: str
    created_at: datetime
    
    class Config:
        from_attributes = True
