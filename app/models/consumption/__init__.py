"""消费分析数据模型初始化

此模块导出所有消费分析相关的数据模型。
"""
from app.models.consumption.gdp_data import GDPData
from app.models.consumption.population_data import PopulationData
from app.models.consumption.economic_indicator import EconomicIndicator
from app.models.consumption.community_data import CommunityData
from app.models.consumption.consumption_forecast import ConsumptionForecast
from app.models.consumption.consumption_category import ConsumptionCategory
from app.models.consumption.consumption_datum import ConsumptionData

__all__ = [
    'GDPData',
    'PopulationData',
    'EconomicIndicator',
    'CommunityData',
    'ConsumptionForecast',
    'ConsumptionCategory',
    'ConsumptionData',
]
