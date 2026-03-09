"""GDP服务

此模块提供GDP相关的业务逻辑。
"""
from typing import List, Optional
from sqlalchemy.orm import Session

from app.models.consumption.gdp_data import GDPData


class GDPService:
    """GDP服务类"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_gdp_data_by_id(self, item_id: int) -> Optional[GDPData]:
        """获取GDP数据"""
        return self.db.query(GDPData).filter(GDPData.id == item_id).first()
    
    def get_gdp_data(
        self,
        skip: int = 0,
        limit: int = 20,
        region_code: Optional[str] = None,
        year: Optional[int] = None
    ) -> List[GDPData]:
        """获取GDP数据列表"""
        query = self.db.query(GDPData)
        
        if region_code:
            query = query.filter(GDPData.region_code == region_code)
        
        if year:
            query = query.filter(GDPData.year == year)
        
        return query.order_by(GDPData.year.desc()).offset(skip).limit(limit).all()
    
    def create_gdp_data(self, data: dict) -> GDPData:
        """创建GDP数据"""
        item = GDPData(**data)
        self.db.add(item)
        self.db.commit()
        self.db.refresh(item)
        return item
    
    def update_gdp_data(self, item_id: int, data: dict) -> Optional[GDPData]:
        """更新GDP数据"""
        item = self.get_gdp_data_by_id(item_id)
        if not item:
            return None
        
        for key, value in data.items():
            if hasattr(item, key) and value is not None:
                setattr(item, key, value)
        
        self.db.commit()
        self.db.refresh(item)
        return item
    
    def delete_gdp_data(self, item_id: int) -> bool:
        """删除GDP数据"""
        item = self.get_gdp_data_by_id(item_id)
        if not item:
            return False
        
        self.db.delete(item)
        self.db.commit()
        return True
