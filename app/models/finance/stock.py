"""股票行情数据模型

此模块定义股票行情数据模型。
"""
from datetime import datetime, date
from sqlalchemy import Column, Integer, String, BigInteger, Float, Date, DateTime, Text, Numeric
from sqlalchemy import Index
from app.core.database import Base


class Stock(Base):
    """股票行情数据模型"""
    
    __tablename__ = "stocks"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    ts_code = Column(String(20), index=True, nullable=False, comment="TS代码")
    trade_date = Column(Date, index=True, nullable=False, comment="交易日期")
    open = Column(Numeric(10, 3), nullable=True, comment="开盘价")
    high = Column(Numeric(10, 3), nullable=True, comment="最高价")
    low = Column(Numeric(10, 3), nullable=True, comment="最低价")
    close = Column(Numeric(10, 3), nullable=True, comment="收盘价")
    pre_close = Column(Numeric(10, 3), nullable=True, comment="昨收价")
    change = Column(Numeric(10, 3), nullable=True, comment="涨跌额")
    pct_chg = Column(Numeric(10, 3), nullable=True, comment="涨跌幅(%)")
    vol = Column(BigInteger, nullable=True, comment="成交量(手)")
    amount = Column(Numeric(20, 3), nullable=True, comment="成交额(千元)")
    
    ma5 = Column(Numeric(10, 3), nullable=True, comment="5日均线")
    ma10 = Column(Numeric(10, 3), nullable=True, comment="10日均线")
    ma20 = Column(Numeric(10, 3), nullable=True, comment="20日均线")
    ma60 = Column(Numeric(10, 3), nullable=True, comment="60日均线")
    
    turnover_rate = Column(Numeric(10, 3), nullable=True, comment="换手率(%)")
    volume_ratio = Column(Numeric(10, 3), nullable=True, comment="量比")
    pe = Column(Numeric(10, 3), nullable=True, comment="市盈率")
    pe_ttm = Column(Numeric(10, 3), nullable=True, comment="市盈率TTM")
    pb = Column(Numeric(10, 3), nullable=True, comment="市净率")
    ps = Column(Numeric(10, 3), nullable=True, comment="市销率")
    total_mv = Column(BigInteger, nullable=True, comment="总市值(万元)")
    circ_mv = Column(BigInteger, nullable=True, comment="流通市值(万元)")
    
    created_at = Column(DateTime, default=datetime.utcnow, comment="创建时间")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, comment="更新时间")
    
    __table_args__ = (
        Index('idx_stocks_ts_code_date', 'ts_code', 'trade_date'),
        Index('idx_stocks_trade_date', 'trade_date'),
    )
    
    def __repr__(self):
        return f"<Stock(id={self.id}, ts_code='{self.ts_code}', trade_date='{self.trade_date}')>"
