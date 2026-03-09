"""八字服务

此模块提供八字相关的业务逻辑。
"""
from typing import List, Optional
from sqlalchemy.orm import Session

from app.models.fortune.bazi import Bazi


class BaziService:
    """八字服务类"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_bazi(self, item_id: int) -> Optional[Bazi]:
        """获取八字数据"""
        return self.db.query(Bazi).filter(Bazi.id == item_id).first()
    
    def get_bazi_list(
        self,
        skip: int = 0,
        limit: int = 20
    ) -> List[Bazi]:
        """获取八字数据列表"""
        return self.db.query(Bazi).offset(skip).limit(limit).all()
    
    def create_bazi(self, data: dict) -> Bazi:
        """创建八字数据"""
        item = Bazi(**data)
        self.db.add(item)
        self.db.commit()
        self.db.refresh(item)
        return item
    
    def update_bazi(self, item_id: int, data: dict) -> Optional[Bazi]:
        """更新八字数据"""
        item = self.get_bazi(item_id)
        if not item:
            return None
        
        for key, value in data.items():
            if hasattr(item, key) and value is not None:
                setattr(item, key, value)
        
        self.db.commit()
        self.db.refresh(item)
        return item
    
    def delete_bazi(self, item_id: int) -> bool:
        """删除八字数据"""
        item = self.get_bazi(item_id)
        if not item:
            return False
        
        self.db.delete(item)
        self.db.commit()
        return True
