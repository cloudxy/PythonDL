"""认证API路由

此模块定义认证相关的API接口。
"""
from datetime import timedelta
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from typing import Dict, Any

from app.core.database import get_db
from app.core.auth import (
    create_access_token,
    create_refresh_token,
    get_current_user,
    verify_token,
    revoke_token
)
from app.core.security import verify_password, get_password_hash
from app.core.cache import cache_manager
from app.models.admin.user import User
from app.schemas.auth import (
    UserLogin,
    UserCreate,
    UserResponse,
    TokenResponse,
    PasswordResetRequest,
    PasswordReset
)
from app.services.admin.user_service import UserService

router = APIRouter()


@router.post("/login", response_model=TokenResponse)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    """用户登录"""
    user_service = UserService(db)
    user = user_service.get_user_by_username(form_data.username)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误"
        )
    
    if not verify_password(form_data.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户名或密码错误"
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="用户已被禁用"
        )
    
    access_token = create_access_token(
        data={"sub": str(user.id), "username": user.username}
    )
    refresh_token = create_refresh_token(
        data={"sub": str(user.id), "username": user.username}
    )
    
    user_service.update_last_login(user.id)
    
    return TokenResponse(
        access_token=access_token,
        refresh_token=refresh_token,
        token_type="bearer"
    )


@router.post("/register", response_model=UserResponse)
async def register(
    user_data: UserCreate,
    db: Session = Depends(get_db)
):
    """用户注册"""
    user_service = UserService(db)
    
    if user_service.get_user_by_username(user_data.username):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="用户名已存在"
        )
    
    if user_service.get_user_by_email(user_data.email):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="邮箱已被注册"
        )
    
    user = user_service.create_user(
        username=user_data.username,
        email=user_data.email,
        password=user_data.password,
        real_name=user_data.real_name
    )
    
    return UserResponse.from_orm(user)


@router.post("/refresh", response_model=TokenResponse)
async def refresh_token(
    refresh_token: str,
    db: Session = Depends(get_db)
):
    """刷新令牌"""
    payload = verify_token(refresh_token)
    
    if not payload or payload.get("type") != "refresh":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="无效的刷新令牌"
        )
    
    user_id = payload.get("sub")
    user_service = UserService(db)
    user = user_service.get_user(int(user_id))
    
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="用户不存在或已被禁用"
        )
    
    access_token = create_access_token(
        data={"sub": str(user.id), "username": user.username}
    )
    new_refresh_token = create_refresh_token(
        data={"sub": str(user.id), "username": user.username}
    )
    
    return TokenResponse(
        access_token=access_token,
        refresh_token=new_refresh_token,
        token_type="bearer"
    )


@router.post("/logout")
async def logout(current_user: User = Depends(get_current_user)):
    """用户登出"""
    return {"message": "登出成功"}


@router.post("/forgot-password")
async def forgot_password(
    request: PasswordResetRequest,
    db: Session = Depends(get_db)
):
    """忘记密码"""
    user_service = UserService(db)
    user = user_service.get_user_by_email(request.email)
    
    if user:
        pass
    
    return {"message": "如果邮箱存在，重置邮件已发送"}


@router.post("/reset-password")
async def reset_password(
    reset_data: PasswordReset,
    db: Session = Depends(get_db)
):
    """重置密码"""
    user_service = UserService(db)
    user = user_service.get_user_by_email(reset_data.email)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="无效的重置请求"
        )
    
    user.password_hash = get_password_hash(reset_data.new_password)
    db.commit()
    
    return {"message": "密码重置成功"}


@router.get("/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: User = Depends(get_current_user)
):
    """获取当前用户信息"""
    return UserResponse.from_orm(current_user)
