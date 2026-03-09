"""股票基础信息模型

此模块定义股票基础信息数据模型。
"""
from datetime import datetime, date
from sqlalchemy import Column, Integer, String, BigInteger, Float, Date, DateTime, Text
from sqlalchemy import Index
from app.core.database import Base


class StockBasic(Base):
    """股票基础信息模型"""
    
    __tablename__ = "stock_basics"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    ts_code = Column(String(20), unique=True, index=True, nullable=False, comment="TS代码")
    symbol = Column(String(10), index=True, nullable=False, comment="股票代码")
    name = Column(String(50), nullable=False, comment="股票名称")
    area = Column(String(20), nullable=True, comment="地域")
    industry = Column(String(50), nullable=True, comment="所属行业")
    market = Column(String(20), nullable=True, comment="市场类型")
    exchange = Column(String(20), nullable=True, comment="交易所代码")
    list_status = Column(String(10), default="L", comment="上市状态")
    list_date = Column(Date, nullable=True, comment="上市日期")
    delist_date = Column(Date, nullable=True, comment="退市日期")
    is_hs = Column(String(10), nullable=True, comment="是否沪深港通标的")
    curr_type = Column(String(10), default="CNY", comment="交易货币")
    created_at = Column(DateTime, default=datetime.utcnow, comment="创建时间")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, comment="更新时间")
    
    __table_args__ = (
        Index('idx_stock_basics_symbol_market', 'symbol', 'market'),
    )
    
    def __repr__(self):
        return f"<StockBasic(id={self.id}, symbol='{self.symbol}', name='{self.name}')>"
