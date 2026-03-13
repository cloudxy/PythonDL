# mypy: ignore-errors
# type: ignore
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from typing import List, Optional
from app.core.deps import get_db
from app.core.auth import get_current_user
from app.admin.services.system_config_service import SystemConfigService
from app.schemas.admin.system_config import SystemConfigCreate, SystemConfigUpdate, SystemConfigResponse

router = APIRouter(prefix="/configs", tags=["系统配置"])


@router.post("", response_model=SystemConfigResponse, summary="创建系统配置")
async def create_config(config_in: SystemConfigCreate, db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)) -> SystemConfigResponse:
    service = SystemConfigService(db)
    existing_config = service.get_config_by_key(config_in.config_key)
    if existing_config:
        raise HTTPException(status_code=400, detail="配置键已存在")
    return service.create_config(config_in)


@router.get("", response_model=List[SystemConfigResponse], summary="获取系统配置列表")
async def get_configs(skip: int = 0, limit: int = 100, config_type: Optional[str] = None, is_active: Optional[bool] = None, 
                db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)) -> List[SystemConfigResponse]:
    service = SystemConfigService(db)
    return service.get_configs(skip=skip, limit=limit, config_type=config_type, is_active=is_active)


@router.get("/{config_id}", response_model=SystemConfigResponse, summary="获取系统配置详情")
async def get_config(config_id: int, db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)) -> SystemConfigResponse:
    service = SystemConfigService(db)
    config = service.get_config(config_id)
    if not config:
        raise HTTPException(status_code=404, detail="配置不存在")
    return config


@router.put("/{config_id}", response_model=SystemConfigResponse, summary="更新系统配置")
async def update_config(config_id: int, config_in: SystemConfigUpdate, db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)) -> SystemConfigResponse:
    service = SystemConfigService(db)
    config = service.update_config(config_id, config_in)
    if not config:
        raise HTTPException(status_code=404, detail="配置不存在")
    return config


@router.delete("/{config_id}", summary="删除系统配置")
async def delete_config(config_id: int, db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)) -> dict:
    service = SystemConfigService(db)
    success = service.delete_config(config_id)
    if not success:
        raise HTTPException(status_code=404, detail="配置不存在")
    return {"message": "删除成功"}


@router.get("/key/{config_key}", summary="根据键获取配置值")
async def get_config_value(config_key: str, default: Optional[str] = None, db: Session = Depends(get_db), current_user: dict = Depends(get_current_user)) -> dict:
    service = SystemConfigService(db)
    value = service.get_config_value(config_key, default)
    if value is None:
        raise HTTPException(status_code=404, detail="配置不存在")
    return {"config_key": config_key, "config_value": value}
