"""股票风险评估服务

此模块提供股票风险评估相关的业务逻辑。
"""
from typing import Dict, Any, Optional
from datetime import date, timedelta
from sqlalchemy.orm import Session
import numpy as np
import pandas as pd
import logging

from app.models.finance.stock import Stock
from app.models.finance.stock_risk_assessment import StockRiskAssessment
from app.core.cache import cache_manager

logger = logging.getLogger(__name__)


class StockRiskService:
    """股票风险评估服务类"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def assess(self, ts_code: str) -> Dict[str, Any]:
        """股票风险评估"""
        cache_key = f"stock_risk:{ts_code}"
        cached_result = cache_manager.get(cache_key)
        if cached_result:
            return cached_result
        
        stock_data = self._get_historical_data(ts_code, days=365)
        
        if stock_data.empty:
            return {
                "ts_code": ts_code,
                "assessment_date": date.today(),
                "risk_score": 50.0,
                "risk_level": "中等",
                "volatility": None,
                "beta": None,
                "sharpe_ratio": None,
                "max_drawdown": None,
                "var_95": None,
                "var_99": None,
                "liquidity_risk": None,
                "market_risk": None,
                "recommendations": "数据不足，无法进行风险评估"
            }
        
        result = {
            "ts_code": ts_code,
            "assessment_date": date.today(),
            "risk_score": self._calculate_risk_score(stock_data),
            "risk_level": self._get_risk_level(self._calculate_risk_score(stock_data)),
            "volatility": self._calculate_volatility(stock_data),
            "beta": self._calculate_beta(stock_data),
            "sharpe_ratio": self._calculate_sharpe_ratio(stock_data),
            "max_drawdown": self._calculate_max_drawdown(stock_data),
            "var_95": self._calculate_var(stock_data, 0.95),
            "var_99": self._calculate_var(stock_data, 0.99),
            "liquidity_risk": self._calculate_liquidity_risk(stock_data),
            "market_risk": self._calculate_market_risk(stock_data),
            "recommendations": self._generate_recommendations(stock_data)
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
                'close': float(stock.close) if stock.close else None,
                'volume': stock.vol
            })
        
        df = pd.DataFrame(data)
        df.set_index('date', inplace=True)
        return df
    
    def _calculate_risk_score(self, data: pd.DataFrame) -> float:
        """计算风险评分"""
        try:
            volatility = self._calculate_volatility(data)
            max_drawdown = self._calculate_max_drawdown(data)
            
            risk_score = (volatility * 50 + abs(max_drawdown) * 0.5)
            return round(min(100, max(0, risk_score)), 2)
        except Exception:
            return 50.0
    
    def _get_risk_level(self, risk_score: float) -> str:
        """获取风险等级"""
        if risk_score < 30:
            return "低风险"
        elif risk_score < 50:
            return "中低风险"
        elif risk_score < 70:
            return "中等风险"
        elif risk_score < 85:
            return "中高风险"
        else:
            return "高风险"
    
    def _calculate_volatility(self, data: pd.DataFrame) -> float:
        """计算波动率"""
        try:
            returns = data['close'].pct_change().dropna()
            return round(returns.std() * np.sqrt(252) * 100, 2)
        except Exception:
            return 0.0
    
    def _calculate_beta(self, data: pd.DataFrame) -> float:
        """计算Beta系数"""
        try:
            returns = data['close'].pct_change().dropna()
            market_var = 0.01
            return round(returns.var() / market_var, 2)
        except Exception:
            return 1.0
    
    def _calculate_sharpe_ratio(self, data: pd.DataFrame) -> float:
        """计算夏普比率"""
        try:
            returns = data['close'].pct_change().dropna()
            risk_free_rate = 0.03
            excess_returns = returns - risk_free_rate / 252
            return round(excess_returns.mean() / returns.std() * np.sqrt(252), 2)
        except Exception:
            return 0.0
    
    def _calculate_max_drawdown(self, data: pd.DataFrame) -> float:
        """计算最大回撤"""
        try:
            cummax = data['close'].cummax()
            drawdown = (data['close'] - cummax) / cummax
            return round(drawdown.min() * 100, 2)
        except Exception:
            return 0.0
    
    def _calculate_var(self, data: pd.DataFrame, confidence: float) -> float:
        """计算VaR"""
        try:
            returns = data['close'].pct_change().dropna()
            var = np.percentile(returns, (1 - confidence) * 100)
            return round(abs(var) * 100, 2)
        except Exception:
            return 0.0
    
    def _calculate_liquidity_risk(self, data: pd.DataFrame) -> float:
        """计算流动性风险"""
        try:
            avg_volume = data['volume'].mean()
            recent_volume = data['volume'].tail(20).mean()
            ratio = recent_volume / avg_volume if avg_volume > 0 else 1
            
            if ratio < 0.5:
                return round(80 + (0.5 - ratio) * 40, 2)
            elif ratio < 1:
                return round(50 + (1 - ratio) * 30, 2)
            else:
                return round(30, 2)
        except Exception:
            return 50.0
    
    def _calculate_market_risk(self, data: pd.DataFrame) -> float:
        """计算市场风险"""
        try:
            volatility = self._calculate_volatility(data)
            beta = self._calculate_beta(data)
            return round((volatility + beta * 10) / 2, 2)
        except Exception:
            return 50.0
    
    def _generate_recommendations(self, data: pd.DataFrame) -> str:
        """生成风险建议"""
        risk_score = self._calculate_risk_score(data)
        
        if risk_score < 30:
            return "该股票风险较低，适合稳健型投资者。建议可以适当增加仓位。"
        elif risk_score < 50:
            return "该股票风险适中，建议保持适度仓位，注意分散投资。"
        elif risk_score < 70:
            return "该股票风险较高，建议控制仓位，设置止损点，谨慎投资。"
        else:
            return "该股票风险很高，建议减少仓位或暂时回避，等待市场稳定。"
