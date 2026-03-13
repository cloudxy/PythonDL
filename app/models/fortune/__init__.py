"""看相算命数据模型初始化

此模块导出所有看相算命相关的数据模型。
"""
from app.models.fortune.feng_shui import FengShui
from app.models.fortune.face_reading import FaceReading
from app.models.fortune.bazi import Bazi
from app.models.fortune.zhou_yi import ZhouYi
from app.models.fortune.constellation import Constellation
from app.models.fortune.fortune_telling import FortuneTelling
from app.models.fortune.face_readings import FaceReadings
from app.models.fortune.fortune_tellings import FortuneTellings

__all__ = [
    'FengShui',
    'FaceReading',
    'Bazi',
    'ZhouYi',
    'Constellation',
    'FortuneTelling',
    'FaceReadings',
    'FortuneTellings',
]
