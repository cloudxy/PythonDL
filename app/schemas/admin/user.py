"""用户 Schema 定义

此模块定义用户管理相关的数据验证 Schema。
"""
from datetime import datetime
from typing import Optional
from pydantic import BaseModel, EmailStr, Field, ConfigDict


class UserBase(BaseModel):
    """用户基础 Schema"""
    username: str = Field(..., min_length=3, max_length=50, description="用户名")
    email: EmailStr = Field(..., description="邮箱")
    real_name: Optional[str] = Field(None, max_length=100, description="真实姓名")


class UserCreate(UserBase):
    """创建用户 Schema"""
    password: str = Field(..., min_length=6, max_length=128, description="密码")
    role_id: Optional[int] = Field(None, description="角色 ID")
    is_active: bool = Field(True, description="是否激活")


class UserUpdate(BaseModel):
    """更新用户 Schema"""
    email: Optional[EmailStr] = Field(None, description="邮箱")
    real_name: Optional[str] = Field(None, max_length=100, description="真实姓名")
    password: Optional[str] = Field(None, min_length=6, max_length=128, description="密码")
    role_id: Optional[int] = Field(None, description="角色 ID")
    is_active: Optional[bool] = Field(None, description="是否激活")


class UserResponse(UserBase):
    """用户响应 Schema"""
    model_config = ConfigDict(from_attributes=True)
    
    id: int
    avatar: Optional[str] = None
    phone: Optional[str] = None
    role_id: Optional[int] = None
    is_active: bool
    is_superuser: bool
    last_login: Optional[datetime] = None
    login_count: int = 0
    created_at: datetime
    updated_at: datetime


class UserListResponse(BaseModel):
    """用户列表响应 Schema"""
    success: bool
    data: dict
    message: str
