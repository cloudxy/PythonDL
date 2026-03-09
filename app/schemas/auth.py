"""认证相关Schema

此模块定义认证相关的数据验证模型。
"""
from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from datetime import datetime


class UserLogin(BaseModel):
    """用户登录"""
    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)


class UserCreate(BaseModel):
    """用户创建"""
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    password: str = Field(..., min_length=6)
    real_name: Optional[str] = Field(None, max_length=50)


class UserResponse(BaseModel):
    """用户响应"""
    id: int
    username: str
    email: str
    real_name: Optional[str]
    avatar: Optional[str]
    role_id: Optional[int]
    is_active: bool
    created_at: datetime
    
    class Config:
        from_attributes = True


class TokenResponse(BaseModel):
    """令牌响应"""
    access_token: str
    refresh_token: str
    token_type: str = "bearer"


class PasswordResetRequest(BaseModel):
    """密码重置请求"""
    email: EmailStr


class PasswordReset(BaseModel):
    """密码重置"""
    email: EmailStr
    token: str
    new_password: str = Field(..., min_length=6)
