"""权限服务

此模块提供权限相关的业务逻辑。
"""
from typing import List, Optional
from sqlalchemy.orm import Session

from app.models.admin.permission import Permission


class PermissionService:
    """权限服务类"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_permission(self, permission_id: int) -> Optional[Permission]:
        """获取权限"""
        return self.db.query(Permission).filter(Permission.id == permission_id).first()
    
    def get_permission_by_code(self, permission_code: str) -> Optional[Permission]:
        """通过权限编码获取权限"""
        return self.db.query(Permission).filter(Permission.permission_code == permission_code).first()
    
    def get_permissions(
        self,
        skip: int = 0,
        limit: int = 20
    ) -> List[Permission]:
        """获取权限列表"""
        return self.db.query(Permission).offset(skip).limit(limit).all()
    
    def create_permission(self, data: dict) -> Permission:
        """创建权限"""
        permission = Permission(**data)
        self.db.add(permission)
        self.db.commit()
        self.db.refresh(permission)
        return permission
    
    def update_permission(self, permission_id: int, data: dict) -> Optional[Permission]:
        """更新权限"""
        permission = self.get_permission(permission_id)
        if not permission:
            return None
        
        for key, value in data.items():
            if hasattr(permission, key) and value is not None:
                setattr(permission, key, value)
        
        self.db.commit()
        self.db.refresh(permission)
        return permission
    
    def delete_permission(self, permission_id: int) -> bool:
        """删除权限"""
        permission = self.get_permission(permission_id)
        if not permission:
            return False
        
        self.db.delete(permission)
        self.db.commit()
        return True
