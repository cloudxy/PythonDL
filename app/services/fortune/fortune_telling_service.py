"""运势服务

此模块提供运势相关的业务逻辑。
"""
from typing import List, Optional
from sqlalchemy.orm import Session

from app.models.fortune.fortune_telling import FortuneTelling


class FortuneTellingService:
    """运势服务类"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_fortune_telling(self, item_id: int) -> Optional[FortuneTelling]:
        """获取运势数据"""
        return self.db.query(FortuneTelling).filter(FortuneTelling.id == item_id).first()
    
    def get_fortune_telling_list(
        self,
        skip: int = 0,
        limit: int = 20,
        category: Optional[str] = None
    ) -> List[FortuneTelling]:
        """获取运势数据列表"""
        query = self.db.query(FortuneTelling)
        
        if category:
            query = query.filter(FortuneTelling.category == category)
        
        return query.offset(skip).limit(limit).all()
    
    def create_fortune_telling(self, data: dict) -> FortuneTelling:
        """创建运势数据"""
        item = FortuneTelling(**data)
        self.db.add(item)
        self.db.commit()
        self.db.refresh(item)
        return item
    
    def update_fortune_telling(self, item_id: int, data: dict) -> Optional[FortuneTelling]:
        """更新运势数据"""
        item = self.get_fortune_telling(item_id)
        if not item:
            return None
        
        for key, value in data.items():
            if hasattr(item, key) and value is not None:
                setattr(item, key, value)
        
        self.db.commit()
        self.db.refresh(item)
        return item
    
    def delete_fortune_telling(self, item_id: int) -> bool:
        """删除运势数据"""
        item = self.get_fortune_telling(item_id)
        if not item:
            return False
        
        self.db.delete(item)
        self.db.commit()
        return True
