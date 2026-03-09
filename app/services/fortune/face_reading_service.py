"""面相服务

此模块提供面相相关的业务逻辑。
"""
from typing import List, Optional
from sqlalchemy.orm import Session

from app.models.fortune.face_reading import FaceReading


class FaceReadingService:
    """面相服务类"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_face_reading(self, item_id: int) -> Optional[FaceReading]:
        """获取面相数据"""
        return self.db.query(FaceReading).filter(FaceReading.id == item_id).first()
    
    def get_face_reading_list(
        self,
        skip: int = 0,
        limit: int = 20,
        face_part: Optional[str] = None
    ) -> List[FaceReading]:
        """获取面相数据列表"""
        query = self.db.query(FaceReading)
        
        if face_part:
            query = query.filter(FaceReading.face_part == face_part)
        
        return query.offset(skip).limit(limit).all()
    
    def create_face_reading(self, data: dict) -> FaceReading:
        """创建面相数据"""
        item = FaceReading(**data)
        self.db.add(item)
        self.db.commit()
        self.db.refresh(item)
        return item
    
    def update_face_reading(self, item_id: int, data: dict) -> Optional[FaceReading]:
        """更新面相数据"""
        item = self.get_face_reading(item_id)
        if not item:
            return None
        
        for key, value in data.items():
            if hasattr(item, key) and value is not None:
                setattr(item, key, value)
        
        self.db.commit()
        self.db.refresh(item)
        return item
    
    def delete_face_reading(self, item_id: int) -> bool:
        """删除面相数据"""
        item = self.get_face_reading(item_id)
        if not item:
            return False
        
        self.db.delete(item)
        self.db.commit()
        return True
