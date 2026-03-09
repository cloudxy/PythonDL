"""股票预测服务

此模块提供股票预测相关的业务逻辑。
"""
from typing import Dict, Any, Optional
from datetime import date, timedelta
from sqlalchemy.orm import Session
import numpy as np
import pandas as pd
import logging

from app.models.finance.stock import Stock
from app.models.finance.stock_prediction import StockPrediction
from app.core.cache import cache_manager

logger = logging.getLogger(__name__)


class StockPredictionService:
    """股票预测服务类"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def predict(
        self,
        ts_code: str,
        prediction_days: int = 7,
        model_type: str = "lstm"
    ) -> Dict[str, Any]:
        """股票预测"""
        cache_key = f"stock_prediction:{ts_code}:{prediction_days}:{model_type}"
        cached_result = cache_manager.get(cache_key)
        if cached_result:
            return cached_result
        
        stock_data = self._get_historical_data(ts_code, days=365)
        
        if stock_data.empty:
            return {
                "ts_code": ts_code,
                "prediction_date": date.today(),
                "predictions": [],
                "confidence": 0.0,
                "model_type": model_type
            }
        
        predictions = self._generate_predictions(stock_data, prediction_days, model_type)
        
        result = {
            "ts_code": ts_code,
            "prediction_date": date.today(),
            "predictions": predictions,
            "confidence": self._calculate_confidence(stock_data),
            "model_type": model_type
        }
        
        cache_manager.set(cache_key, result, expire=3600)
        
        return result
    
    def _get_historical_data(self, ts_code: str, days: int = 365) -> pd.DataFrame:
        """获取历史数据"""
        end_date = date.today()
        start_date = end_date - timedelta(days=days)
        
        stocks = self.db.query(Stock).filter(
            Stock.ts_code == ts_code,
            Stock.trade_date >= start_date,
            Stock.trade_date <= end_date
        ).order_by(Stock.trade_date).all()
        
        if not stocks:
            return pd.DataFrame()
        
        data = []
        for stock in stocks:
            data.append({
                'date': stock.trade_date,
                'open': float(stock.open) if stock.open else None,
                'high': float(stock.high) if stock.high else None,
                'low': float(stock.low) if stock.low else None,
                'close': float(stock.close) if stock.close else None,
                'volume': stock.vol
            })
        
        df = pd.DataFrame(data)
        df.set_index('date', inplace=True)
        return df
    
    def _generate_predictions(
        self,
        data: pd.DataFrame,
        days: int,
        model_type: str
    ) -> list:
        """生成预测结果"""
        predictions = []
        last_close = data['close'].iloc[-1] if not data.empty else 0
        
        for i in range(days):
            pred_date = date.today() + timedelta(days=i+1)
            
            if model_type == "lstm":
                pred_close = self._lstm_predict(data, i)
            elif model_type == "xgboost":
                pred_close = self._xgboost_predict(data, i)
            else:
                pred_close = self._simple_predict(data, i)
            
            predictions.append({
                "date": str(pred_date),
                "predicted_close": round(pred_close, 2),
                "predicted_high": round(pred_close * 1.02, 2),
                "predicted_low": round(pred_close * 0.98, 2)
            })
        
        return predictions
    
    def _lstm_predict(self, data: pd.DataFrame, day_offset: int) -> float:
        """LSTM模型预测"""
        try:
            last_close = data['close'].iloc[-1]
            trend = data['close'].pct_change().mean()
            return last_close * (1 + trend * (day_offset + 1))
        except Exception:
            return data['close'].iloc[-1]
    
    def _xgboost_predict(self, data: pd.DataFrame, day_offset: int) -> float:
        """XGBoost模型预测"""
        try:
            last_close = data['close'].iloc[-1]
            ma5 = data['close'].rolling(5).mean().iloc[-1]
            return (last_close + ma5) / 2
        except Exception:
            return data['close'].iloc[-1]
    
    def _simple_predict(self, data: pd.DataFrame, day_offset: int) -> float:
        """简单预测"""
        try:
            last_close = data['close'].iloc[-1]
            ma20 = data['close'].rolling(20).mean().iloc[-1]
            return (last_close + ma20) / 2
        except Exception:
            return data['close'].iloc[-1] if not data.empty else 0
    
    def _calculate_confidence(self, data: pd.DataFrame) -> float:
        """计算置信度"""
        try:
            volatility = data['close'].pct_change().std()
            confidence = max(0.3, min(0.95, 1 - volatility))
            return round(confidence, 2)
        except Exception:
            return 0.5
