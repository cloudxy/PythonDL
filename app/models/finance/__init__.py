"""金融数据模型初始化

此模块导出所有金融相关的数据模型。
"""
from app.models.finance.stock_basic import StockBasic
from app.models.finance.stock import Stock
from app.models.finance.stock_prediction import StockPrediction
from app.models.finance.stock_risk_assessment import StockRiskAssessment

__all__ = [
    'StockBasic',
    'Stock',
    'StockPrediction',
    'StockRiskAssessment',
]
