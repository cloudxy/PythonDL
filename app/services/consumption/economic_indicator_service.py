"""经济指标服务

此模块提供经济指标相关的业务逻辑。
"""
from typing import List, Optional
from sqlalchemy.orm import Session

from app.models.consumption.economic_indicator import EconomicIndicator


class EconomicIndicatorService:
    """经济指标服务类"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_economic_indicator(self, item_id: int) -> Optional[EconomicIndicator]:
        """获取经济指标数据"""
        return self.db.query(EconomicIndicator).filter(EconomicIndicator.id == item_id).first()
    
    def get_economic_indicators(
        self,
        skip: int = 0,
        limit: int = 20,
        region_code: Optional[str] = None,
        year: Optional[int] = None
    ) -> List[EconomicIndicator]:
        """获取经济指标数据列表"""
        query = self.db.query(EconomicIndicator)
        
        if region_code:
            query = query.filter(EconomicIndicator.region_code == region_code)
        
        if year:
            query = query.filter(EconomicIndicator.year == year)
        
        return query.order_by(EconomicIndicator.year.desc(), EconomicIndicator.month.desc()).offset(skip).limit(limit).all()
    
    def create_economic_indicator(self, data: dict) -> EconomicIndicator:
        """创建经济指标数据"""
        item = EconomicIndicator(**data)
        self.db.add(item)
        self.db.commit()
        self.db.refresh(item)
        return item
