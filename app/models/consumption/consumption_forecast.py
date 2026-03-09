"""消费预测数据模型

此模块定义消费预测数据模型。
"""
from datetime import datetime, date
from sqlalchemy import Column, Integer, String, Float, Date, DateTime, Text, Numeric, BigInteger
from sqlalchemy import Index
from app.core.database import Base


class ConsumptionForecast(Base):
    """消费预测数据模型"""
    
    __tablename__ = "consumption_forecasts"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    region_code = Column(String(20), index=True, nullable=False, comment="区域代码")
    region_name = Column(String(100), nullable=False, comment="区域名称")
    
    forecast_date = Column(Date, index=True, nullable=False, comment="预测日期")
    target_date = Column(Date, nullable=False, comment="目标日期")
    target_period = Column(String(20), nullable=True, comment="目标周期")
    
    predicted_gdp = Column(BigInteger, nullable=True, comment="预测GDP(万元)")
    predicted_gdp_growth = Column(Numeric(10, 3), nullable=True, comment="预测GDP增长率(%)")
    
    predicted_retail_sales = Column(BigInteger, nullable=True, comment="预测零售总额(万元)")
    predicted_retail_growth = Column(Numeric(10, 3), nullable=True, comment="预测零售增长率(%)")
    
    predicted_cpi = Column(Numeric(10, 3), nullable=True, comment="预测CPI")
    predicted_inflation = Column(Numeric(10, 3), nullable=True, comment="预测通胀率(%)")
    
    predicted_consumption_level = Column(Numeric(15, 2), nullable=True, comment="预测消费水平(元)")
    predicted_consumption_growth = Column(Numeric(10, 3), nullable=True, comment="预测消费增长率(%)")
    
    confidence = Column(Numeric(5, 3), nullable=True, comment="置信度")
    model_type = Column(String(50), nullable=True, comment="模型类型")
    model_version = Column(String(50), nullable=True, comment="模型版本")
    
    prediction_params = Column(Text, nullable=True, comment="预测参数")
    features_used = Column(Text, nullable=True, comment="使用的特征")
    
    actual_gdp = Column(BigInteger, nullable=True, comment="实际GDP(万元)")
    actual_retail_sales = Column(BigInteger, nullable=True, comment="实际零售总额(万元)")
    prediction_error = Column(Numeric(10, 3), nullable=True, comment="预测误差(%)")
    
    status = Column(String(20), default="pending", comment="状态")
    
    description = Column(Text, nullable=True, comment="描述")
    recommendations = Column(Text, nullable=True, comment="建议")
    
    created_at = Column(DateTime, default=datetime.utcnow, comment="创建时间")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, comment="更新时间")
    
    __table_args__ = (
        Index('idx_consumption_forecasts_region_date', 'region_code', 'forecast_date'),
    )
    
    def __repr__(self):
        return f"<ConsumptionForecast(id={self.id}, region_name='{self.region_name}', target_date='{self.target_date}')>"
