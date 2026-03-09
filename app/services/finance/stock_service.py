"""股票服务

此模块提供股票相关的业务逻辑。
"""
from typing import List, Optional
from datetime import date
from sqlalchemy.orm import Session

from app.models.finance.stock_basic import StockBasic
from app.models.finance.stock import Stock


class StockService:
    """股票服务类"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_stock_basic(self, stock_id: int) -> Optional[StockBasic]:
        """获取股票基础信息"""
        return self.db.query(StockBasic).filter(StockBasic.id == stock_id).first()
    
    def get_stock_basic_by_code(self, ts_code: str) -> Optional[StockBasic]:
        """通过TS代码获取股票基础信息"""
        return self.db.query(StockBasic).filter(StockBasic.ts_code == ts_code).first()
    
    def get_stock_basics(
        self,
        skip: int = 0,
        limit: int = 20,
        symbol: Optional[str] = None,
        industry: Optional[str] = None
    ) -> List[StockBasic]:
        """获取股票基础信息列表"""
        query = self.db.query(StockBasic)
        
        if symbol:
            query = query.filter(StockBasic.symbol.ilike(f"%{symbol}%"))
        
        if industry:
            query = query.filter(StockBasic.industry == industry)
        
        return query.offset(skip).limit(limit).all()
    
    def create_stock_basic(self, data: dict) -> StockBasic:
        """创建股票基础信息"""
        stock = StockBasic(**data)
        self.db.add(stock)
        self.db.commit()
        self.db.refresh(stock)
        return stock
    
    def update_stock_basic(self, stock_id: int, data: dict) -> Optional[StockBasic]:
        """更新股票基础信息"""
        stock = self.get_stock_basic(stock_id)
        if not stock:
            return None
        
        for key, value in data.items():
            if hasattr(stock, key) and value is not None:
                setattr(stock, key, value)
        
        self.db.commit()
        self.db.refresh(stock)
        return stock
    
    def delete_stock_basic(self, stock_id: int) -> bool:
        """删除股票基础信息"""
        stock = self.get_stock_basic(stock_id)
        if not stock:
            return False
        
        self.db.delete(stock)
        self.db.commit()
        return True
    
    def get_stock_data(
        self,
        skip: int = 0,
        limit: int = 20,
        ts_code: Optional[str] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None
    ) -> List[Stock]:
        """获取股票行情数据"""
        query = self.db.query(Stock)
        
        if ts_code:
            query = query.filter(Stock.ts_code == ts_code)
        
        if start_date:
            query = query.filter(Stock.trade_date >= start_date)
        
        if end_date:
            query = query.filter(Stock.trade_date <= end_date)
        
        return query.order_by(Stock.trade_date.desc()).offset(skip).limit(limit).all()
    
    def create_stock_data(self, data: dict) -> Stock:
        """创建股票行情数据"""
        stock_data = Stock(**data)
        self.db.add(stock_data)
        self.db.commit()
        self.db.refresh(stock_data)
        return stock_data
    
    def batch_create_stock_data(self, data_list: List[dict]) -> int:
        """批量创建股票行情数据"""
        count = 0
        for data in data_list:
            stock_data = Stock(**data)
            self.db.add(stock_data)
            count += 1
        self.db.commit()
        return count
