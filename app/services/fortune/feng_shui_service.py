"""风水服务

此模块提供风水相关的业务逻辑。
"""
from typing import List, Optional
from sqlalchemy.orm import Session

from app.models.fortune.feng_shui import FengShui


class FengShuiService:
    """风水服务类"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_feng_shui(self, item_id: int) -> Optional[FengShui]:
        """获取风水数据"""
        return self.db.query(FengShui).filter(FengShui.id == item_id).first()
    
    def get_feng_shui_list(
        self,
        skip: int = 0,
        limit: int = 20,
        category: Optional[str] = None
    ) -> List[FengShui]:
        """获取风水数据列表"""
        query = self.db.query(FengShui)
        
        if category:
            query = query.filter(FengShui.category == category)
        
        return query.offset(skip).limit(limit).all()
    
    def create_feng_shui(self, data: dict) -> FengShui:
        """创建风水数据"""
        item = FengShui(**data)
        self.db.add(item)
        self.db.commit()
        self.db.refresh(item)
        return item
    
    def update_feng_shui(self, item_id: int, data: dict) -> Optional[FengShui]:
        """更新风水数据"""
        item = self.get_feng_shui(item_id)
        if not item:
            return None
        
        for key, value in data.items():
            if hasattr(item, key) and value is not None:
                setattr(item, key, value)
        
        self.db.commit()
        self.db.refresh(item)
        return item
    
    def delete_feng_shui(self, item_id: int) -> bool:
        """删除风水数据"""
        item = self.get_feng_shui(item_id)
        if not item:
            return False
        
        self.db.delete(item)
        self.db.commit()
        return True
