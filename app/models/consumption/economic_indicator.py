"""经济指标数据模型

此模块定义经济指标数据模型。
"""
from datetime import datetime, date
from sqlalchemy import Column, Integer, String, Float, Date, DateTime, Text, Numeric, BigInteger
from sqlalchemy import Index
from app.core.database import Base


class EconomicIndicator(Base):
    """经济指标数据模型"""
    
    __tablename__ = "economic_indicators"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    region_code = Column(String(20), index=True, nullable=False, comment="区域代码")
    region_name = Column(String(100), nullable=False, comment="区域名称")
    region_level = Column(String(20), nullable=True, comment="区域级别")
    
    year = Column(Integer, index=True, nullable=False, comment="年份")
    month = Column(Integer, nullable=True, comment="月份")
    
    cpi = Column(Numeric(10, 3), nullable=True, comment="消费者物价指数")
    cpi_yoy = Column(Numeric(10, 3), nullable=True, comment="CPI同比(%)")
    cpi_mom = Column(Numeric(10, 3), nullable=True, comment="CPI环比(%)")
    
    ppi = Column(Numeric(10, 3), nullable=True, comment="生产者物价指数")
    ppi_yoy = Column(Numeric(10, 3), nullable=True, comment="PPI同比(%)")
    ppi_mom = Column(Numeric(10, 3), nullable=True, comment="PPI环比(%)")
    
    pmi = Column(Numeric(10, 3), nullable=True, comment="采购经理指数")
    pmi_manufacturing = Column(Numeric(10, 3), nullable=True, comment="制造业PMI")
    pmi_non_manufacturing = Column(Numeric(10, 3), nullable=True, comment="非制造业PMI")
    
    retail_sales = Column(BigInteger, nullable=True, comment="社会消费品零售总额(万元)")
    retail_sales_yoy = Column(Numeric(10, 3), nullable=True, comment="零售总额同比(%)")
    
    fixed_asset_investment = Column(BigInteger, nullable=True, comment="固定资产投资(万元)")
    fai_yoy = Column(Numeric(10, 3), nullable=True, comment="固定资产投资同比(%)")
    
    industrial_value_added = Column(BigInteger, nullable=True, comment="工业增加值(万元)")
    iva_yoy = Column(Numeric(10, 3), nullable=True, comment="工业增加值同比(%)")
    
    export_value = Column(BigInteger, nullable=True, comment="出口总额(万美元)")
    import_value = Column(BigInteger, nullable=True, comment="进口总额(万美元)")
    trade_balance = Column(BigInteger, nullable=True, comment="贸易顺差(万美元)")
    
    fdi = Column(BigInteger, nullable=True, comment="外商直接投资(万美元)")
    
    unemployment_rate = Column(Numeric(10, 3), nullable=True, comment="失业率(%)")
    
    m2 = Column(BigInteger, nullable=True, comment="M2货币供应量(亿元)")
    m1 = Column(BigInteger, nullable=True, comment="M1货币供应量(亿元)")
    m0 = Column(BigInteger, nullable=True, comment="M0货币供应量(亿元)")
    
    data_source = Column(String(100), nullable=True, comment="数据来源")
    data_quality = Column(String(20), default="official", comment="数据质量")
    
    description = Column(Text, nullable=True, comment="描述")
    
    created_at = Column(DateTime, default=datetime.utcnow, comment="创建时间")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, comment="更新时间")
    
    __table_args__ = (
        Index('idx_economic_indicators_region_year', 'region_code', 'year'),
    )
    
    def __repr__(self):
        return f"<EconomicIndicator(id={self.id}, region_name='{self.region_name}', year={self.year})>"
