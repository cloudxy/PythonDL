# type: ignore
from typing import List, Optional
from sqlalchemy.orm import Session
from app.models.system import SystemConfig
from app.schemas.admin.system_config import SystemConfigCreate, SystemConfigUpdate
from datetime import datetime


class SystemConfigService:
    def __init__(self, db: Session):
        self.db = db

    def create_config(self, config_in: SystemConfigCreate) -> SystemConfig:
        try:
            config = SystemConfig(
                config_key=config_in.config_key,
                config_value=config_in.config_value,
                config_type=config_in.config_type,
                description=config_in.description,
                is_active=config_in.is_active,
                created_at=datetime.now(),
                updated_at=datetime.now()
            )
            self.db.add(config)
            self.db.commit()
            self.db.refresh(config)
            return config
        except Exception:
            self.db.rollback()
            raise

    def get_config(self, config_id: int) -> Optional[SystemConfig]:
        return self.db.query(SystemConfig).filter(SystemConfig.id == config_id).first()

    def get_config_by_key(self, config_key: str) -> Optional[SystemConfig]:
        return self.db.query(SystemConfig).filter(SystemConfig.config_key == config_key).first()

    def get_configs(self, skip: int = 0, limit: int = 100, config_type: str = None, is_active: bool = None) -> List[SystemConfig]:
        query = self.db.query(SystemConfig)
        if config_type:
            query = query.filter(SystemConfig.config_type == config_type)
        if is_active is not None:
            query = query.filter(SystemConfig.is_active == is_active)
        return query.offset(skip).limit(limit).all()

    def update_config(self, config_id: int, config_in: SystemConfigUpdate) -> Optional[SystemConfig]:
        try:
            config = self.get_config(config_id)
            if not config:
                return None
            
            update_data = config_in.model_dump(exclude_unset=True)
            for field, value in update_data.items():
                setattr(config, field, value)
            
            config.updated_at = datetime.now()
            self.db.commit()
            self.db.refresh(config)
            return config
        except Exception:
            self.db.rollback()
            raise

    def delete_config(self, config_id: int) -> bool:
        try:
            config = self.get_config(config_id)
            if not config:
                return False
            
            self.db.delete(config)
            self.db.commit()
            return True
        except Exception:
            self.db.rollback()
            raise

    def get_config_value(self, config_key: str, default: str = None) -> Optional[str]:
        config = self.get_config_by_key(config_key)
        return config.config_value if config else default
