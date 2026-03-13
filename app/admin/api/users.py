"""用户 API 接口

此模块定义用户管理相关的 API 接口。
"""
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.core.auth import get_current_user
from app.models.admin.user import User
from app.admin.services.user_service import UserService
from app.schemas.admin.user import (
    UserCreate,
    UserUpdate,
    UserResponse,
    UserListResponse
)

router = APIRouter(prefix="/users", tags=["用户管理"])


@router.get("", response_model=UserListResponse)
async def get_users(
    skip: int = 0,
    limit: int = 20,
    username: Optional[str] = None,
    is_active: Optional[bool] = None,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """获取用户列表"""
    user_service = UserService(db)
    users = user_service.get_users(
        skip=skip,
        limit=limit,
        username=username,
        is_active=is_active
    )
    total = user_service.count_users()
    
    return {
        "success": True,
        "data": {
            "items": users,
            "total": total,
            "skip": skip,
            "limit": limit
        },
        "message": "获取成功"
    }


@router.post("", response_model=UserResponse)
async def create_user(
    user_data: UserCreate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """创建用户"""
    user_service = UserService(db)
    
    # 检查用户名是否存在
    if user_service.get_user_by_username(user_data.username):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="用户名已存在"
        )
    
    # 检查邮箱是否存在
    if user_service.get_user_by_email(user_data.email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="邮箱已被注册"
        )
    
    user = user_service.create_user(
        username=user_data.username,
        email=user_data.email,
        password=user_data.password,
        real_name=user_data.real_name,
        role_id=user_data.role_id,
        is_active=user_data.is_active
    )
    
    return {
        "success": True,
        "data": user,
        "message": "创建成功"
    }


@router.get("/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """获取用户详情"""
    user_service = UserService(db)
    user = user_service.get_user(user_id)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="用户不存在"
        )
    
    return {
        "success": True,
        "data": user,
        "message": "获取成功"
    }


@router.put("/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: int,
    user_data: UserUpdate,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """更新用户"""
    user_service = UserService(db)
    
    user = user_service.update_user(user_id, user_data.model_dump(exclude_unset=True))
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="用户不存在"
        )
    
    return {
        "success": True,
        "data": user,
        "message": "更新成功"
    }


@router.delete("/{user_id}")
async def delete_user(
    user_id: int,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user)
):
    """删除用户"""
    user_service = UserService(db)
    
    if not user_service.delete_user(user_id):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="用户不存在"
        )
    
    return {
        "success": True,
        "message": "删除成功"
    }
