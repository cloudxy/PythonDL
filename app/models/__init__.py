"""模型初始化模块

此模块导出所有数据模型。
"""
from app.models.admin.user import User
from app.models.admin.role import Role
from app.models.admin.permission import Permission
from app.models.admin.role_permission import RolePermission
from app.models.admin.system_config import SystemConfig
from app.models.admin.operation_log import OperationLog

from app.models.finance.stock_basic import StockBasic
from app.models.finance.stock import Stock
from app.models.finance.stock_prediction import StockPrediction
from app.models.finance.stock_risk_assessment import StockRiskAssessment

from app.models.weather.weather_station import WeatherStation
from app.models.weather.weather import Weather
from app.models.weather.weather_forecast import WeatherForecast

from app.models.fortune.feng_shui import FengShui
from app.models.fortune.face_reading import FaceReading
from app.models.fortune.bazi import Bazi
from app.models.fortune.zhou_yi import ZhouYi
from app.models.fortune.constellation import Constellation
from app.models.fortune.fortune_telling import FortuneTelling

from app.models.consumption.gdp_data import GDPData
from app.models.consumption.population_data import PopulationData
from app.models.consumption.economic_indicator import EconomicIndicator
from app.models.consumption.community_data import CommunityData
from app.models.consumption.consumption_forecast import ConsumptionForecast

__all__ = [
    'User',
    'Role',
    'Permission',
    'RolePermission',
    'SystemConfig',
    'OperationLog',
    'StockBasic',
    'Stock',
    'StockPrediction',
    'StockRiskAssessment',
    'WeatherStation',
    'Weather',
    'WeatherForecast',
    'FengShui',
    'FaceReading',
    'Bazi',
    'ZhouYi',
    'Constellation',
    'FortuneTelling',
    'GDPData',
    'PopulationData',
    'EconomicIndicator',
    'CommunityData',
    'ConsumptionForecast',
]
