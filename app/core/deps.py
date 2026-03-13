"""依赖注入模块

此模块提供常用的依赖注入函数。
"""
from typing import Generator

from fastapi import Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.core.database import get_db
from app.models.admin.user import User
from app.core.auth import get_current_user


def get_db_session() -> Generator[Session, None, None]:
    """获取数据库会话（简化版本）
    
    Yields:
        Session: 数据库会话
    """
    db = next(get_db())
    try:
        yield db
    finally:
        pass


async def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """获取当前活跃用户
    
    Args:
        current_user: 当前认证用户
        
    Returns:
        User: 活跃用户对象
    """
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="User account is inactive"
        )
    return current_user


async def get_current_superuser(
    current_user: User = Depends(get_current_active_user),
) -> User:
    """获取当前超级管理员
    
    Args:
        current_user: 当前活跃用户
        
    Returns:
        User: 超级管理员对象
    """
    if not current_user.is_superuser:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="The user doesn't have enough privileges"
        )
    return current_user
