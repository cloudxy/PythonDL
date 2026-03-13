"""股票风险评估服务

此模块提供股票风险评估功能。
"""
from typing import Dict, Optional
import numpy as np
from datetime import datetime, timedelta
from sqlalchemy.orm import Session

from app.models.finance.stock import Stock


class StockRiskService:
    """股票风险评估服务类"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def assess_risk(self, ts_code: str, days: int = 60) -> Optional[Dict]:
        """
        评估股票风险
        
        Args:
            ts_code: 股票代码
            days: 评估天数
            
        Returns:
            风险评估结果
        """
        # 获取历史数据
        stocks = self._get_recent_stocks(ts_code, days)
        
        if not stocks or len(stocks) < 30:
            return None
        
        # 提取收盘价
        close_prices = [stock.close for stock in stocks]
        
        # 计算波动率
        volatility = self._calculate_volatility(close_prices)
        
        # 计算 VaR (Value at Risk)
        var_95 = self._calculate_var(close_prices, confidence=0.95)
        
        # 计算夏普比率
        sharpe_ratio = self._calculate_sharpe_ratio(close_prices)
        
        # 计算最大回撤
        max_drawdown = self._calculate_max_drawdown(close_prices)
        
        # 综合风险评分 (0-100, 越高风险越大)
        risk_score = self._calculate_risk_score(
            volatility=volatility,
            var_95=var_95,
            sharpe_ratio=sharpe_ratio,
            max_drawdown=max_drawdown
        )
        
        risk_level = self._get_risk_level(risk_score)
        
        return {
            "ts_code": ts_code,
            "risk_score": round(risk_score, 2),
            "risk_level": risk_level,
            "volatility": round(volatility, 4),
            "var_95": round(var_95, 4),
            "sharpe_ratio": round(sharpe_ratio, 2),
            "max_drawdown": round(max_drawdown, 4),
            "assessment_date": datetime.now().strftime("%Y-%m-%d")
        }
    
    def _get_recent_stocks(self, ts_code: str, days: int = 60) -> list:
        """获取最近 N 天的股票数据"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        return self.db.query(Stock).filter(
            Stock.ts_code == ts_code,
            Stock.trade_date >= start_date
        ).order_by(Stock.trade_date.asc()).all()
    
    def _calculate_volatility(self, prices: list) -> float:
        """计算波动率"""
        returns = np.diff(prices) / prices[:-1]
        return float(np.std(returns)) * np.sqrt(252)  # 年化波动率
    
    def _calculate_var(self, prices: list, confidence: float = 0.95) -> float:
        """计算 VaR"""
        returns = np.diff(prices) / prices[:-1]
        return float(np.percentile(returns, (1 - confidence) * 100))
    
    def _calculate_sharpe_ratio(self, prices: list) -> float:
        """计算夏普比率"""
        returns = np.diff(prices) / prices[:-1]
        if np.std(returns) == 0:
            return 0
        return float(np.mean(returns) / np.std(returns)) * np.sqrt(252)
    
    def _calculate_max_drawdown(self, prices: list) -> float:
        """计算最大回撤"""
        peak = prices[0]
        max_dd = 0
        
        for price in prices:
            if price > peak:
                peak = price
            drawdown = (peak - price) / peak
            if drawdown > max_dd:
                max_dd = drawdown
        
        return max_dd
    
    def _calculate_risk_score(
        self,
        volatility: float,
        var_95: float,
        sharpe_ratio: float,
        max_drawdown: float
    ) -> float:
        """计算综合风险评分"""
        # 波动率权重 40%
        vol_score = min(volatility * 100, 100) * 0.4
        
        # VaR 权重 30%
        var_score = min(abs(var_95) * 1000, 100) * 0.3
        
        # 夏普比率权重 10% (越低分越高)
        sharpe_score = max(0, 100 - sharpe_ratio * 20) * 0.1
        
        # 最大回撤权重 20%
        drawdown_score = min(max_drawdown * 100, 100) * 0.2
        
        return vol_score + var_score + sharpe_score + drawdown_score
    
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
