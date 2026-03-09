"""角色服务

此模块提供角色相关的业务逻辑。
"""
from typing import List, Optional
from sqlalchemy.orm import Session

from app.models.admin.role import Role


class RoleService:
    """角色服务类"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_role(self, role_id: int) -> Optional[Role]:
        """获取角色"""
        return self.db.query(Role).filter(Role.id == role_id).first()
    
    def get_role_by_code(self, role_code: str) -> Optional[Role]:
        """通过角色编码获取角色"""
        return self.db.query(Role).filter(Role.role_code == role_code).first()
    
    def get_roles(
        self,
        skip: int = 0,
        limit: int = 20
    ) -> List[Role]:
        """获取角色列表"""
        return self.db.query(Role).offset(skip).limit(limit).all()
    
    def create_role(self, data: dict) -> Role:
        """创建角色"""
        role = Role(**data)
        self.db.add(role)
        self.db.commit()
        self.db.refresh(role)
        return role
    
    def update_role(self, role_id: int, data: dict) -> Optional[Role]:
        """更新角色"""
        role = self.get_role(role_id)
        if not role:
            return None
        
        for key, value in data.items():
            if hasattr(role, key) and value is not None:
                setattr(role, key, value)
        
        self.db.commit()
        self.db.refresh(role)
        return role
    
    def delete_role(self, role_id: int) -> bool:
        """删除角色"""
        role = self.get_role(role_id)
        if not role:
            return False
        
        self.db.delete(role)
        self.db.commit()
        return True
