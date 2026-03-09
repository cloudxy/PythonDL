"""用户服务

此模块提供用户相关的业务逻辑。
"""
from typing import List, Optional
from datetime import datetime
from sqlalchemy.orm import Session
from sqlalchemy import or_

from app.models.admin.user import User
from app.core.security import get_password_hash


class UserService:
    """用户服务类"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_user(self, user_id: int) -> Optional[User]:
        """获取用户"""
        return self.db.query(User).filter(User.id == user_id).first()
    
    def get_user_by_username(self, username: str) -> Optional[User]:
        """通过用户名获取用户"""
        return self.db.query(User).filter(User.username == username).first()
    
    def get_user_by_email(self, email: str) -> Optional[User]:
        """通过邮箱获取用户"""
        return self.db.query(User).filter(User.email == email).first()
    
    def get_users(
        self,
        skip: int = 0,
        limit: int = 20,
        username: Optional[str] = None
    ) -> List[User]:
        """获取用户列表"""
        query = self.db.query(User)
        
        if username:
            query = query.filter(User.username.ilike(f"%{username}%"))
        
        return query.offset(skip).limit(limit).all()
    
    def create_user(
        self,
        username: str,
        email: str,
        password: str,
        real_name: Optional[str] = None,
        role_id: Optional[int] = None
    ) -> User:
        """创建用户"""
        user = User(
            username=username,
            email=email,
            password_hash=get_password_hash(password),
            real_name=real_name,
            role_id=role_id
        )
        self.db.add(user)
        self.db.commit()
        self.db.refresh(user)
        return user
    
    def update_user(self, user_id: int, data: dict) -> Optional[User]:
        """更新用户"""
        user = self.get_user(user_id)
        if not user:
            return None
        
        for key, value in data.items():
            if hasattr(user, key) and value is not None:
                setattr(user, key, value)
        
        self.db.commit()
        self.db.refresh(user)
        return user
    
    def delete_user(self, user_id: int) -> bool:
        """删除用户"""
        user = self.get_user(user_id)
        if not user:
            return False
        
        self.db.delete(user)
        self.db.commit()
        return True
    
    def update_last_login(self, user_id: int) -> bool:
        """更新最后登录时间"""
        user = self.get_user(user_id)
        if not user:
            return False
        
        user.last_login = datetime.utcnow()
        user.login_count = (user.login_count or 0) + 1
        self.db.commit()
        return True
    
    def count_users(self) -> int:
        """统计用户总数"""
        return self.db.query(User).count()
    
    def count_active_users(self) -> int:
        """统计活跃用户数"""
        return self.db.query(User).filter(User.is_active == True).count()
