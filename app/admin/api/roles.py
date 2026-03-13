# mypy: ignore-errors
# type: ignore
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional
from app.core.deps import get_db
from app.core.auth import get_current_user
from app.admin.services.role_service import RoleService
from app.schemas.admin.role import RoleCreate, RoleUpdate, RoleResponse

router = APIRouter(prefix="/roles", tags=["角色管理"])


@router.post("", response_model=RoleResponse, summary="创建角色")
async def create_role(role_in: RoleCreate, db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)) -> RoleResponse:
    service = RoleService(db)
    existing_role = service.get_role_by_code(role_in.code)
    if existing_role:
        raise HTTPException(status_code=400, detail="角色编码已存在")
    return service.create_role(role_in)


@router.get("", response_model=List[RoleResponse], summary="获取角色列表")
async def get_roles(skip: int = 0, limit: int = 100, is_active: Optional[bool] = None, 
              db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)) -> List[RoleResponse]:
    service = RoleService(db)
    return service.get_roles(skip=skip, limit=limit, is_active=is_active)


@router.get("/{role_id}", response_model=RoleResponse, summary="获取角色详情")
async def get_role(role_id: int, db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)) -> RoleResponse:
    service = RoleService(db)
    role = service.get_role(role_id)
    if not role:
        raise HTTPException(status_code=404, detail="角色不存在")
    return role


@router.put("/{role_id}", response_model=RoleResponse, summary="更新角色")
async def update_role(role_id: int, role_in: RoleUpdate, db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)) -> RoleResponse:
    service = RoleService(db)
    role = service.update_role(role_id, role_in)
    if not role:
        raise HTTPException(status_code=404, detail="角色不存在")
    return role


@router.delete("/{role_id}", summary="删除角色")
async def delete_role(role_id: int, db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)) -> dict:
    service = RoleService(db)
    success = service.delete_role(role_id)
    if not success:
        raise HTTPException(status_code=404, detail="角色不存在")
    return {"message": "删除成功"}


@router.post("/{role_id}/users/{user_id}", summary="分配角色给用户")
async def assign_role_to_user(role_id: int, user_id: int, db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)) -> dict:
    service = RoleService(db)
    try:
        return service.assign_role_to_user(user_id, role_id)
    except Exception:
        raise HTTPException(status_code=400, detail="分配失败")


@router.delete("/{role_id}/users/{user_id}", summary="从用户移除角色")
async def remove_role_from_user(role_id: int, user_id: int, db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)) -> dict:
    service = RoleService(db)
    success = service.remove_role_from_user(user_id, role_id)
    if not success:
        raise HTTPException(status_code=404, detail="角色未分配给该用户")
    return {"message": "移除成功"}


@router.get("/{role_id}/permissions", summary="获取角色权限")
async def get_role_permissions(role_id: int, db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)) -> List:
    service = RoleService(db)
    return service.get_role_permissions(role_id)
