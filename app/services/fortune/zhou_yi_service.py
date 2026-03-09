"""周易服务

此模块提供周易相关的业务逻辑。
"""
from typing import List, Optional
from sqlalchemy.orm import Session

from app.models.fortune.zhou_yi import ZhouYi


class ZhouYiService:
    """周易服务类"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_zhou_yi(self, item_id: int) -> Optional[ZhouYi]:
        """获取周易数据"""
        return self.db.query(ZhouYi).filter(ZhouYi.id == item_id).first()
    
    def get_zhou_yi_by_number(self, hexagram_number: int) -> Optional[ZhouYi]:
        """通过卦序获取周易数据"""
        return self.db.query(ZhouYi).filter(ZhouYi.hexagram_number == hexagram_number).first()
    
    def get_zhou_yi_list(
        self,
        skip: int = 0,
        limit: int = 20
    ) -> List[ZhouYi]:
        """获取周易数据列表"""
        return self.db.query(ZhouYi).order_by(ZhouYi.hexagram_number).offset(skip).limit(limit).all()
    
    def create_zhou_yi(self, data: dict) -> ZhouYi:
        """创建周易数据"""
        item = ZhouYi(**data)
        self.db.add(item)
        self.db.commit()
        self.db.refresh(item)
        return item
