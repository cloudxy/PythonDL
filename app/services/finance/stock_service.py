"""股票服务

此模块提供股票相关的业务逻辑。
"""
from typing import List, Optional, Tuple
from datetime import datetime, timedelta
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import and_, func, select
import json

from app.models.finance.stock import Stock
from app.models.finance.stock_basic import StockBasic
from app.core.redis_client import redis_client


class StockService:
    """股票服务类"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.redis = redis_client
        self.cache_prefix = "stock:"
        self.cache_ttl = 300  # 5 分钟缓存
    
    async def get_stock(self, stock_id: int) -> Optional[Stock]:
        """获取股票"""
        # 尝试从缓存获取
        cache_key = f"{self.cache_prefix}id:{stock_id}"
        cached = await self.redis.get(cache_key)
        if cached:
            return cached
        
        # 从数据库获取
        result = await self.db.execute(
            select(Stock).where(Stock.id == stock_id)
        )
        stock = result.scalar_one_or_none()
        
        # 写入缓存
        if stock:
            await self.redis.set(cache_key, stock.__dict__, expire=self.cache_ttl)
        
        return stock
    
    async def get_stock_by_code(self, ts_code: str) -> Optional[Stock]:
        """通过代码获取股票"""
        # 尝试从缓存获取
        cache_key = f"{self.cache_prefix}code:{ts_code}"
        cached = await self.redis.get(cache_key)
        if cached:
            return cached
        
        # 从数据库获取
        result = await self.db.execute(
            select(Stock).where(Stock.ts_code == ts_code)
        )
        stock = result.scalar_one_or_none()
        
        # 写入缓存
        if stock:
            await self.redis.set(cache_key, stock.__dict__, expire=self.cache_ttl)
        
        return stock
    
    async def get_stocks(
        self,
        page: int = 1,
        page_size: int = 20,
        ts_code: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Tuple[List[Stock], int]:
        """获取股票列表"""
        # 构建查询条件
        conditions = []
        if ts_code:
            conditions.append(Stock.ts_code.ilike(f"%{ts_code}%"))
        if start_date:
            conditions.append(Stock.trade_date >= start_date)
        if end_date:
            conditions.append(Stock.trade_date <= end_date)
        
        # 查询总数
        if conditions:
            count_query = select(func.count()).select_from(Stock).where(and_(*conditions))
        else:
            count_query = select(func.count()).select_from(Stock)
        
        total_result = await self.db.execute(count_query)
        total = total_result.scalar() or 0
        
        # 分页查询
        query = select(Stock)
        if conditions:
            query = query.where(and_(*conditions))
        query = query.order_by(Stock.trade_date.desc())
        query = query.offset((page - 1) * page_size).limit(page_size)
        
        result = await self.db.execute(query)
        stocks = result.scalars().all()
        
        return list(stocks), total
    
    def create_stock(self, data: dict) -> Stock:
        """创建股票记录"""
        try:
            stock = Stock(**data)
            self.db.add(stock)
            self.db.commit()
            self.db.refresh(stock)
            return stock
        except Exception:
            self.db.rollback()
            raise
    
    def update_stock(self, stock_id: int, data: dict) -> Optional[Stock]:
        """更新股票"""
        try:
            stock = self.get_stock(stock_id)
            if not stock:
                return None
            
            for key, value in data.items():
                if hasattr(stock, key) and value is not None:
                    setattr(stock, key, value)
            
            self.db.commit()
            self.db.refresh(stock)
            return stock
        except Exception:
            self.db.rollback()
            raise
    
    def delete_stock(self, stock_id: int) -> bool:
        """删除股票"""
        try:
            stock = self.get_stock(stock_id)
            if not stock:
                return False
            
            self.db.delete(stock)
            self.db.commit()
            return True
        except Exception:
            self.db.rollback()
            raise
    
    def get_recent_stocks(self, ts_code: str, days: int = 30) -> List[Stock]:
        """获取指定股票近 N 天的数据"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        return self.db.query(Stock).filter(
            and_(
                Stock.ts_code == ts_code,
                Stock.trade_date >= start_date,
                Stock.trade_date <= end_date
            )
        ).order_by(Stock.trade_date.desc()).all()
    
    def count_stocks(self) -> int:
        """统计股票总数"""
        return self.db.query(Stock).count()
