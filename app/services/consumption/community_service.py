"""小区服务

此模块提供小区相关的业务逻辑。
"""
from typing import List, Optional
from sqlalchemy.orm import Session

from app.models.consumption.community_data import CommunityData


class CommunityService:
    """小区服务类"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_community_data_by_id(self, item_id: int) -> Optional[CommunityData]:
        """获取小区数据"""
        return self.db.query(CommunityData).filter(CommunityData.id == item_id).first()
    
    def get_community_data(
        self,
        skip: int = 0,
        limit: int = 20,
        city: Optional[str] = None,
        district: Optional[str] = None
    ) -> List[CommunityData]:
        """获取小区数据列表"""
        query = self.db.query(CommunityData)
        
        if city:
            query = query.filter(CommunityData.city == city)
        
        if district:
            query = query.filter(CommunityData.district == district)
        
        return query.offset(skip).limit(limit).all()
    
    def create_community_data(self, data: dict) -> CommunityData:
        """创建小区数据"""
        item = CommunityData(**data)
        self.db.add(item)
        self.db.commit()
        self.db.refresh(item)
        return item
    
    def update_community_data(self, item_id: int, data: dict) -> Optional[CommunityData]:
        """更新小区数据"""
        item = self.get_community_data_by_id(item_id)
        if not item:
            return None
        
        for key, value in data.items():
            if hasattr(item, key) and value is not None:
                setattr(item, key, value)
        
        self.db.commit()
        self.db.refresh(item)
        return item
    
    def delete_community_data(self, item_id: int) -> bool:
        """删除小区数据"""
        item = self.get_community_data_by_id(item_id)
        if not item:
            return False
        
        self.db.delete(item)
        self.db.commit()
        return True
