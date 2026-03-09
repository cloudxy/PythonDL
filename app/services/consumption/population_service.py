"""人口服务

此模块提供人口相关的业务逻辑。
"""
from typing import List, Optional
from sqlalchemy.orm import Session

from app.models.consumption.population_data import PopulationData


class PopulationService:
    """人口服务类"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_population_data_by_id(self, item_id: int) -> Optional[PopulationData]:
        """获取人口数据"""
        return self.db.query(PopulationData).filter(PopulationData.id == item_id).first()
    
    def get_population_data(
        self,
        skip: int = 0,
        limit: int = 20,
        region_code: Optional[str] = None,
        year: Optional[int] = None
    ) -> List[PopulationData]:
        """获取人口数据列表"""
        query = self.db.query(PopulationData)
        
        if region_code:
            query = query.filter(PopulationData.region_code == region_code)
        
        if year:
            query = query.filter(PopulationData.year == year)
        
        return query.order_by(PopulationData.year.desc()).offset(skip).limit(limit).all()
    
    def create_population_data(self, data: dict) -> PopulationData:
        """创建人口数据"""
        item = PopulationData(**data)
        self.db.add(item)
        self.db.commit()
        self.db.refresh(item)
        return item
    
    def update_population_data(self, item_id: int, data: dict) -> Optional[PopulationData]:
        """更新人口数据"""
        item = self.get_population_data_by_id(item_id)
        if not item:
            return None
        
        for key, value in data.items():
            if hasattr(item, key) and value is not None:
                setattr(item, key, value)
        
        self.db.commit()
        self.db.refresh(item)
        return item
    
    def delete_population_data(self, item_id: int) -> bool:
        """删除人口数据"""
        item = self.get_population_data_by_id(item_id)
        if not item:
            return False
        
        self.db.delete(item)
        self.db.commit()
        return True
