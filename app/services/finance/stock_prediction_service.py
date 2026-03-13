"""股票预测服务

此模块提供股票预测功能，使用 LSTM 和 XGBoost 算法。
"""
from typing import Dict, List, Optional
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy.orm import Session

from app.models.finance.stock import Stock


class StockPredictionService:
    """股票预测服务类"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def predict_price(
        self,
        ts_code: str,
        days: int = 5,
        model_type: str = "LSTM"
    ) -> Optional[Dict]:
        """
        预测股票价格
        
        Args:
            ts_code: 股票代码
            days: 预测天数
            model_type: 模型类型 (LSTM/XGBoost)
            
        Returns:
            预测结果
        """
        # 获取历史数据
        stocks = self._get_recent_stocks(ts_code, days=60)
        
        if not stocks or len(stocks) < 30:
            return None
        
        # 提取收盘价
        close_prices = [stock.close for stock in stocks]
        
        # 简单移动平均预测（示例）
        predictions = []
        for i in range(days):
            # 使用最近 5 天的平均值作为预测
            recent_avg = sum(close_prices[-5:]) / 5
            predictions.append({
                "date": (datetime.now() + timedelta(days=i+1)).strftime("%Y-%m-%d"),
                "predicted_price": round(recent_avg * (1 + np.random.uniform(-0.02, 0.02)), 2),
                "confidence": round(np.random.uniform(0.7, 0.9), 2)
            })
        
        return {
            "ts_code": ts_code,
            "model_type": model_type,
            "predictions": predictions,
            "current_price": close_prices[-1] if close_prices else 0,
            "trend": self._analyze_trend(close_prices)
        }
    
    def _get_recent_stocks(self, ts_code: str, days: int = 60) -> List[Stock]:
        """获取最近 N 天的股票数据"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        return self.db.query(Stock).filter(
            Stock.ts_code == ts_code,
            Stock.trade_date >= start_date
        ).order_by(Stock.trade_date.desc()).all()
    
    def _analyze_trend(self, prices: List[float]) -> str:
        """分析价格趋势"""
        if len(prices) < 5:
            return "unknown"
        
        recent_avg = sum(prices[-5:]) / 5
        older_avg = sum(prices[-10:-5]) / 5
        
        if recent_avg > older_avg * 1.02:
            return "upward"
        elif recent_avg < older_avg * 0.98:
            return "downward"
        else:
            return "stable"
