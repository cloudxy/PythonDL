# mypy: ignore-errors
# type: ignore
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional
from app.core.deps import get_db
from app.core.auth import get_current_user
from app.admin.services.permission_service import PermissionService
from app.schemas.admin.permission import PermissionCreate, PermissionUpdate, PermissionResponse

router = APIRouter(prefix="/permissions", tags=["权限管理"])


@router.post("", response_model=PermissionResponse, summary="创建权限")
async def create_permission(permission_in: PermissionCreate, db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)) -> PermissionResponse:
    service = PermissionService(db)
    existing_permission = service.get_permission_by_code(permission_in.code)
    if existing_permission:
        raise HTTPException(status_code=400, detail="权限编码已存在")
    return service.create_permission(permission_in)


@router.get("", response_model=List[PermissionResponse], summary="获取权限列表")
async def get_permissions(skip: int = 0, limit: int = 100, resource: Optional[str] = None, 
                    db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)) -> List[PermissionResponse]:
    service = PermissionService(db)
    return service.get_permissions(skip=skip, limit=limit, resource=resource)


@router.get("/{permission_id}", response_model=PermissionResponse, summary="获取权限详情")
async def get_permission(permission_id: int, db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)) -> PermissionResponse:
    service = PermissionService(db)
    permission = service.get_permission(permission_id)
    if not permission:
        raise HTTPException(status_code=404, detail="权限不存在")
    return permission


@router.put("/{permission_id}", response_model=PermissionResponse, summary="更新权限")
async def update_permission(permission_id: int, permission_in: PermissionUpdate, db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)) -> PermissionResponse:
    service = PermissionService(db)
    permission = service.update_permission(permission_id, permission_in)
    if not permission:
        raise HTTPException(status_code=404, detail="权限不存在")
    return permission


@router.delete("/{permission_id}", summary="删除权限")
async def delete_permission(permission_id: int, db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)) -> dict:
    service = PermissionService(db)
    success = service.delete_permission(permission_id)
    if not success:
        raise HTTPException(status_code=404, detail="权限不存在")
    return {"message": "删除成功"}


@router.post("/roles/{role_id}/permissions/{permission_id}", summary="分配权限给角色")
async def assign_permission_to_role(role_id: int, permission_id: int, db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)) -> dict:
    service = PermissionService(db)
    try:
        return service.assign_permission_to_role(role_id, permission_id)
    except Exception:
        raise HTTPException(status_code=400, detail="分配失败")


@router.delete("/roles/{role_id}/permissions/{permission_id}", summary="从角色移除权限")
async def remove_permission_from_role(role_id: int, permission_id: int, db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)) -> dict:
    service = PermissionService(db)
    success = service.remove_permission_from_role(role_id, permission_id)
    if not success:
        raise HTTPException(status_code=404, detail="权限未分配给该角色")
    return {"message": "移除成功"}


@router.get("/users/{user_id}", summary="获取用户权限")
async def get_user_permissions(user_id: int, db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)) -> List[PermissionResponse]:
    service = PermissionService(db)
    return service.get_user_permissions(user_id)
