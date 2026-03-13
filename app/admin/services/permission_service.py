# type: ignore
from typing import List, Optional
from sqlalchemy.orm import Session
from app.models.rbac import Permission, RolePermission, UserRole
from app.schemas.admin.permission import PermissionCreate, PermissionUpdate
from datetime import datetime


class PermissionService:
    def __init__(self, db: Session):
        self.db = db

    def create_permission(self, permission_in: PermissionCreate) -> Permission:
        try:
            permission = Permission(
                name=permission_in.name,
                code=permission_in.code,
                resource=permission_in.resource,
                action=permission_in.action,
                description=permission_in.description,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            self.db.add(permission)
            self.db.commit()
            self.db.refresh(permission)
            return permission
        except Exception:
            self.db.rollback()
            raise

    def get_permission(self, permission_id: int) -> Optional[Permission]:
        return self.db.query(Permission).filter(Permission.id == permission_id).first()

    def get_permission_by_code(self, code: str) -> Optional[Permission]:
        return self.db.query(Permission).filter(Permission.code == code).first()

    def get_permissions(self, skip: int = 0, limit: int = 100, resource: str = None) -> List[Permission]:
        query = self.db.query(Permission)
        if resource:
            query = query.filter(Permission.resource == resource)
        return query.offset(skip).limit(limit).all()

    def update_permission(self, permission_id: int, permission_in: PermissionUpdate) -> Optional[Permission]:
        try:
            permission = self.get_permission(permission_id)
            if not permission:
                return None
            
            update_data = permission_in.model_dump(exclude_unset=True)
            for field, value in update_data.items():
                setattr(permission, field, value)
            
            permission.updated_at = datetime.now()
            self.db.commit()
            self.db.refresh(permission)
            return permission
        except Exception:
            self.db.rollback()
            raise

    def delete_permission(self, permission_id: int) -> bool:
        try:
            permission = self.get_permission(permission_id)
            if not permission:
                return False
            
            self.db.query(RolePermission).filter(RolePermission.permission_id == permission_id).delete()
            self.db.delete(permission)
            self.db.commit()
            return True
        except Exception:
            self.db.rollback()
            raise

    def assign_permission_to_role(self, role_id: int, permission_id: int) -> RolePermission:
        try:
            role_permission = RolePermission(role_id=role_id, permission_id=permission_id)
            self.db.add(role_permission)
            self.db.commit()
            self.db.refresh(role_permission)
            return role_permission
        except Exception:
            self.db.rollback()
            raise

    def remove_permission_from_role(self, role_id: int, permission_id: int) -> bool:
        try:
            role_permission = self.db.query(RolePermission).filter(
                RolePermission.role_id == role_id,
                RolePermission.permission_id == permission_id
            ).first()
            if not role_permission:
                return False
            self.db.delete(role_permission)
            self.db.commit()
            return True
        except Exception:
            self.db.rollback()
            raise

    def get_role_permissions(self, role_id: int) -> List[Permission]:
        role_permissions = self.db.query(RolePermission).filter(RolePermission.role_id == role_id).all()
        permission_ids = [rp.permission_id for rp in role_permissions]
        return self.db.query(Permission).filter(Permission.id.in_(permission_ids)).all()

    def get_user_permissions(self, user_id: int) -> List[Permission]:
        user_roles = self.db.query(UserRole).filter(UserRole.user_id == user_id).all()
        role_ids = [ur.role_id for ur in user_roles]
        role_permissions = self.db.query(RolePermission).filter(RolePermission.role_id.in_(role_ids)).all()
        permission_ids = [rp.permission_id for rp in role_permissions]
        return self.db.query(Permission).filter(Permission.id.in_(permission_ids)).all()
