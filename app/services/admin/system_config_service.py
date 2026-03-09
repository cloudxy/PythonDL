"""系统配置服务

此模块提供系统配置相关的业务逻辑。
"""
from typing import List, Optional
from sqlalchemy.orm import Session

from app.models.admin.system_config import SystemConfig


class SystemConfigService:
    """系统配置服务类"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_config(self, config_id: int) -> Optional[SystemConfig]:
        """获取配置"""
        return self.db.query(SystemConfig).filter(SystemConfig.id == config_id).first()
    
    def get_config_by_key(self, config_key: str) -> Optional[SystemConfig]:
        """通过配置键获取配置"""
        return self.db.query(SystemConfig).filter(SystemConfig.config_key == config_key).first()
    
    def get_configs(
        self,
        skip: int = 0,
        limit: int = 20,
        category: Optional[str] = None
    ) -> List[SystemConfig]:
        """获取配置列表"""
        query = self.db.query(SystemConfig)
        
        if category:
            query = query.filter(SystemConfig.category == category)
        
        return query.offset(skip).limit(limit).all()
    
    def create_config(self, data: dict) -> SystemConfig:
        """创建配置"""
        config = SystemConfig(**data)
        self.db.add(config)
        self.db.commit()
        self.db.refresh(config)
        return config
    
    def update_config(self, config_id: int, data: dict) -> Optional[SystemConfig]:
        """更新配置"""
        config = self.get_config(config_id)
        if not config:
            return None
        
        for key, value in data.items():
            if hasattr(config, key) and value is not None:
                setattr(config, key, value)
        
        self.db.commit()
        self.db.refresh(config)
        return config
    
    def delete_config(self, config_id: int) -> bool:
        """删除配置"""
        config = self.get_config(config_id)
        if not config:
            return False
        
        self.db.delete(config)
        self.db.commit()
        return True
    
    def get_config_value(self, config_key: str, default: str = None) -> Optional[str]:
        """获取配置值"""
        config = self.get_config_by_key(config_key)
        if config:
            return config.config_value
        return default
