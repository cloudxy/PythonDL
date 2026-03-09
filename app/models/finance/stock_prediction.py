"""股票预测模型

此模块定义股票预测数据模型。
"""
from datetime import datetime, date
from sqlalchemy import Column, Integer, String, BigInteger, Float, Date, DateTime, Text, Numeric
from sqlalchemy import Index
from app.core.database import Base


class StockPrediction(Base):
    """股票预测模型"""
    
    __tablename__ = "stock_predictions"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    ts_code = Column(String(20), index=True, nullable=False, comment="TS代码")
    prediction_date = Column(Date, index=True, nullable=False, comment="预测日期")
    target_date = Column(Date, nullable=False, comment="目标日期")
    
    predicted_close = Column(Numeric(10, 3), nullable=True, comment="预测收盘价")
    predicted_high = Column(Numeric(10, 3), nullable=True, comment="预测最高价")
    predicted_low = Column(Numeric(10, 3), nullable=True, comment="预测最低价")
    predicted_change = Column(Numeric(10, 3), nullable=True, comment="预测涨跌幅(%)")
    
    confidence = Column(Numeric(5, 3), nullable=True, comment="置信度")
    model_type = Column(String(50), nullable=True, comment="模型类型")
    model_version = Column(String(50), nullable=True, comment="模型版本")
    
    features_used = Column(Text, nullable=True, comment="使用的特征")
    prediction_params = Column(Text, nullable=True, comment="预测参数")
    
    actual_close = Column(Numeric(10, 3), nullable=True, comment="实际收盘价")
    actual_change = Column(Numeric(10, 3), nullable=True, comment="实际涨跌幅(%)")
    prediction_error = Column(Numeric(10, 3), nullable=True, comment="预测误差(%)")
    
    status = Column(String(20), default="pending", comment="状态")
    created_at = Column(DateTime, default=datetime.utcnow, comment="创建时间")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, comment="更新时间")
    
    __table_args__ = (
        Index('idx_stock_predictions_code_date', 'ts_code', 'prediction_date'),
    )
    
    def __repr__(self):
        return f"<StockPrediction(id={self.id}, ts_code='{self.ts_code}', target_date='{self.target_date}')>"
