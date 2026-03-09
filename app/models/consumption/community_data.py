"""小区数据模型

此模块定义小区数据模型。
"""
from datetime import datetime, date
from sqlalchemy import Column, Integer, String, Float, Date, DateTime, Text, Numeric, BigInteger
from sqlalchemy import Index
from app.core.database import Base


class CommunityData(Base):
    """小区数据模型"""
    
    __tablename__ = "community_data"
    
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    community_id = Column(String(50), unique=True, index=True, nullable=False, comment="小区ID")
    community_name = Column(String(200), nullable=False, comment="小区名称")
    
    province = Column(String(50), nullable=True, comment="省份")
    city = Column(String(50), nullable=True, comment="城市")
    district = Column(String(50), nullable=True, comment="区县")
    street = Column(String(100), nullable=True, comment="街道")
    
    address = Column(String(500), nullable=True, comment="详细地址")
    
    latitude = Column(Numeric(10, 6), nullable=True, comment="纬度")
    longitude = Column(Numeric(10, 6), nullable=True, comment="经度")
    
    developer = Column(String(200), nullable=True, comment="开发商")
    property_company = Column(String(200), nullable=True, comment="物业公司")
    
    build_year = Column(Integer, nullable=True, comment="建成年份")
    total_buildings = Column(Integer, nullable=True, comment="总栋数")
    total_units = Column(Integer, nullable=True, comment="总户数")
    
    building_type = Column(String(50), nullable=True, comment="建筑类型")
    property_type = Column(String(50), nullable=True, comment="物业类型")
    
    greening_rate = Column(Numeric(10, 3), nullable=True, comment="绿化率(%)")
    plot_ratio = Column(Numeric(10, 3), nullable=True, comment="容积率")
    
    property_fee = Column(Numeric(10, 2), nullable=True, comment="物业费(元/平米/月)")
    parking_spaces = Column(Integer, nullable=True, comment="车位数量")
    
    avg_price = Column(Numeric(15, 2), nullable=True, comment="均价(元/平米)")
    price_change = Column(Numeric(10, 3), nullable=True, comment="价格变化(%)")
    
    facilities = Column(Text, nullable=True, comment="配套设施(JSON)")
    nearby_schools = Column(Text, nullable=True, comment="周边学校(JSON)")
    nearby_hospitals = Column(Text, nullable=True, comment="周边医院(JSON)")
    nearby_malls = Column(Text, nullable=True, comment="周边商场(JSON)")
    
    transport = Column(Text, nullable=True, comment="交通信息(JSON)")
    
    description = Column(Text, nullable=True, comment="描述")
    
    data_source = Column(String(100), nullable=True, comment="数据来源")
    data_quality = Column(String(20), default="official", comment="数据质量")
    
    is_active = Column(Integer, default=1, comment="是否活跃")
    created_at = Column(DateTime, default=datetime.utcnow, comment="创建时间")
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow, comment="更新时间")
    
    __table_args__ = (
        Index('idx_community_data_city_district', 'city', 'district'),
    )
    
    def __repr__(self):
        return f"<CommunityData(id={self.id}, community_name='{self.community_name}')>"
