"""星座服务

此模块提供星座相关的业务逻辑。
"""
from typing import List, Optional
from sqlalchemy.orm import Session

from app.models.fortune.constellation import Constellation


class ConstellationService:
    """星座服务类"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_constellation(self, item_id: int) -> Optional[Constellation]:
        """获取星座数据"""
        return self.db.query(Constellation).filter(Constellation.id == item_id).first()
    
    def get_constellation_by_name(self, name: str) -> Optional[Constellation]:
        """通过名称获取星座数据"""
        return self.db.query(Constellation).filter(Constellation.name == name).first()
    
    def get_constellation_list(
        self,
        skip: int = 0,
        limit: int = 20
    ) -> List[Constellation]:
        """获取星座数据列表"""
        return self.db.query(Constellation).offset(skip).limit(limit).all()
    
    def create_constellation(self, data: dict) -> Constellation:
        """创建星座数据"""
        item = Constellation(**data)
        self.db.add(item)
        self.db.commit()
        self.db.refresh(item)
        return item
