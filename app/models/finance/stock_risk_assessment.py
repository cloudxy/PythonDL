"""股票风险评估模型

此模块定义股票风险评估数据模型。
"""
from datetime import datetime, date
from sqlalchemy import Column, Integer, String, BigInteger, Float, Date, DateTime, Text, Numeric
from sqlalchemy import Index
from app.core.database import Base


class StockRiskAssessment(Base):
    """股票风险评估模型"""
    
    __tablename__ = "stock_risk_assessments"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    ts_code = Column(String(20), index=True, nullable=False, comment="TS代码")
    assessment_date = Column(Date, index=True, nullable=False, comment="评估日期")
    
    risk_score = Column(Numeric(5, 2), nullable=True, comment="风险评分(0-100)")
    risk_level = Column(String(20), nullable=True, comment="风险等级")
    
    volatility = Column(Numeric(10, 3), nullable=True, comment="波动率")
    beta = Column(Numeric(10, 3), nullable=True, comment="Beta系数")
    sharpe_ratio = Column(Numeric(10, 3), nullable=True, comment="夏普比率")
    max_drawdown = Column(Numeric(10, 3), nullable=True, comment="最大回撤(%)")
    
    var_95 = Column(Numeric(10, 3), nullable=True, comment="VaR(95%)")
    var_99 = Column(Numeric(10, 3), nullable=True, comment="VaR(99%)")
    
    liquidity_risk = Column(Numeric(5, 2), nullable=True, comment="流动性风险")
    market_risk = Column(Numeric(5, 2), nullable=True, comment="市场风险")
    operational_risk = Column(Numeric(5, 2), nullable=True, comment="操作风险")
    
    assessment_method = Column(String(50), nullable=True, comment="评估方法")
    assessment_params = Column(Text, nullable=True, comment="评估参数")
    risk_factors = Column(Text, nullable=True, comment="风险因素")
    recommendations = Column(Text, nullable=True, comment="风险建议")
    
    status = Column(String(20), default="completed", comment="状态")
    created_at = Column(DateTime, default=datetime.utcnow, comment="创建时间")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, comment="更新时间")
    
    __table_args__ = (
        Index('idx_stock_risk_code_date', 'ts_code', 'assessment_date'),
    )
    
    def __repr__(self):
        return f"<StockRiskAssessment(id={self.id}, ts_code='{self.ts_code}', risk_level='{self.risk_level}')>"
