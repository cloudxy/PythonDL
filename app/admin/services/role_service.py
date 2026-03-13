# type: ignore
from typing import List, Optional
from sqlalchemy.orm import Session
from app.models.rbac import Role, UserRole
from app.schemas.admin.role import RoleCreate, RoleUpdate
from datetime import datetime


class RoleService:
    def __init__(self, db: Session):
        self.db = db

    def create_role(self, role_in: RoleCreate) -> Role:
        try:
            role = Role(
                name=role_in.name,
                code=role_in.code,
                description=role_in.description,
                is_active=role_in.is_active,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            self.db.add(role)
            self.db.commit()
            self.db.refresh(role)
            return role
        except Exception:
            self.db.rollback()
            raise

    def get_role(self, role_id: int) -> Optional[Role]:
        return self.db.query(Role).filter(Role.id == role_id).first()

    def get_role_by_code(self, code: str) -> Optional[Role]:
        return self.db.query(Role).filter(Role.code == code).first()

    def get_roles(self, skip: int = 0, limit: int = 100, is_active: bool = None) -> List[Role]:
        query = self.db.query(Role)
        if is_active is not None:
            query = query.filter(Role.is_active == is_active)
        return query.offset(skip).limit(limit).all()

    def update_role(self, role_id: int, role_in: RoleUpdate) -> Optional[Role]:
        try:
            role = self.get_role(role_id)
            if not role:
                return None
            
            update_data = role_in.model_dump(exclude_unset=True)
            for field, value in update_data.items():
                setattr(role, field, value)
            
            role.updated_at = datetime.now()
            self.db.commit()
            self.db.refresh(role)
            return role
        except Exception:
            self.db.rollback()
            raise

    def delete_role(self, role_id: int) -> bool:
        try:
            role = self.get_role(role_id)
            if not role:
                return False
            
            self.db.query(UserRole).filter(UserRole.role_id == role_id).delete()
            self.db.delete(role)
            self.db.commit()
            return True
        except Exception:
            self.db.rollback()
            raise

    def assign_role_to_user(self, user_id: int, role_id: int) -> UserRole:
        try:
            user_role = UserRole(user_id=user_id, role_id=role_id)
            self.db.add(user_role)
            self.db.commit()
            self.db.refresh(user_role)
            return user_role
        except Exception:
            self.db.rollback()
            raise

    def remove_role_from_user(self, user_id: int, role_id: int) -> bool:
        try:
            user_role = self.db.query(UserRole).filter(
                UserRole.user_id == user_id,
                UserRole.role_id == role_id
            ).first()
            if not user_role:
                return False
            self.db.delete(user_role)
            self.db.commit()
            return True
        except Exception:
            self.db.rollback()
            raise

    def get_user_roles(self, user_id: int) -> List[Role]:
        user_roles = self.db.query(UserRole).filter(UserRole.user_id == user_id).all()
        role_ids = [ur.role_id for ur in user_roles]
        return self.db.query(Role).filter(Role.id.in_(role_ids)).all()
