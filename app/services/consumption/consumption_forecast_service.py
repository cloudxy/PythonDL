"""消费预测服务

此模块提供消费预测相关的业务逻辑。
"""
from typing import Dict, Any
from datetime import date, timedelta
from sqlalchemy.orm import Session
import numpy as np
import pandas as pd
import logging

from app.models.consumption.gdp_data import GDPData
from app.models.consumption.economic_indicator import EconomicIndicator
from app.core.cache import cache_manager

logger = logging.getLogger(__name__)


class ConsumptionForecastService:
    """消费预测服务类"""
    
    def __init__(self, db: Session):
        self.db = db
    
    def forecast(self, region_code: str, forecast_months: int = 12) -> Dict[str, Any]:
        """消费预测"""
        cache_key = f"consumption_forecast:{region_code}:{forecast_months}"
        cached_result = cache_manager.get(cache_key)
        if cached_result:
            return cached_result
        
        gdp_data = self._get_gdp_data(region_code)
        economic_data = self._get_economic_data(region_code)
        
        if gdp_data.empty and economic_data.empty:
            return {
                "region_code": region_code,
                "region_name": "未知区域",
                "forecast_date": date.today(),
                "forecasts": [],
                "confidence": 0.0
            }
        
        forecasts = self._generate_forecasts(gdp_data, economic_data, forecast_months)
        region_name = self._get_region_name(region_code)
        
        result = {
            "region_code": region_code,
            "region_name": region_name,
            "forecast_date": date.today(),
            "forecasts": forecasts,
            "confidence": self._calculate_confidence(gdp_data, economic_data)
        }
        
        cache_manager.set(cache_key, result, expire=3600)
        
        return result
    
    def _get_gdp_data(self, region_code: str) -> pd.DataFrame:
        """获取GDP数据"""
        gdp_records = self.db.query(GDPData).filter(
            GDPData.region_code == region_code
        ).order_by(GDPData.year).all()
        
        if not gdp_records:
            return pd.DataFrame()
        
        data = []
        for record in gdp_records:
            data.append({
                'year': record.year,
                'gdp': float(record.gdp) if record.gdp else None,
                'gdp_growth': float(record.gdp_growth) if record.gdp_growth else None
            })
        
        return pd.DataFrame(data)
    
    def _get_economic_data(self, region_code: str) -> pd.DataFrame:
        """获取经济指标数据"""
        economic_records = self.db.query(EconomicIndicator).filter(
            EconomicIndicator.region_code == region_code
        ).order_by(EconomicIndicator.year, EconomicIndicator.month).all()
        
        if not economic_records:
            return pd.DataFrame()
        
        data = []
        for record in economic_records:
            data.append({
                'year': record.year,
                'month': record.month,
                'cpi': float(record.cpi) if record.cpi else None,
                'retail_sales': float(record.retail_sales) if record.retail_sales else None
            })
        
        return pd.DataFrame(data)
    
    def _get_region_name(self, region_code: str) -> str:
        """获取区域名称"""
        gdp = self.db.query(GDPData).filter(GDPData.region_code == region_code).first()
        return gdp.region_name if gdp else "未知区域"
    
    def _generate_forecasts(
        self,
        gdp_data: pd.DataFrame,
        economic_data: pd.DataFrame,
        months: int
    ) -> list:
        """生成预测结果"""
        forecasts = []
        
        for i in range(months):
            target_date = date.today() + timedelta(days=30 * (i + 1))
            
            forecast = {
                "date": str(target_date),
                "predicted_gdp_growth": self._predict_gdp_growth(gdp_data, i),
                "predicted_retail_growth": self._predict_retail_growth(economic_data, i),
                "predicted_cpi": self._predict_cpi(economic_data, i)
            }
            
            forecasts.append(forecast)
        
        return forecasts
    
    def _predict_gdp_growth(self, data: pd.DataFrame, month_offset: int) -> float:
        """预测GDP增长率"""
        try:
            if data.empty or 'gdp_growth' not in data.columns:
                return 5.0
            
            last_growth = data['gdp_growth'].iloc[-1]
            ma3 = data['gdp_growth'].rolling(3).mean().iloc[-1]
            return round((last_growth + ma3) / 2, 2)
        except Exception:
            return 5.0
    
    def _predict_retail_growth(self, data: pd.DataFrame, month_offset: int) -> float:
        """预测零售增长率"""
        try:
            if data.empty or 'retail_sales' not in data.columns:
                return 8.0
            
            last_sales = data['retail_sales'].iloc[-1]
            prev_sales = data['retail_sales'].iloc[-2] if len(data) > 1 else last_sales
            
            if prev_sales and prev_sales > 0:
                growth = (last_sales - prev_sales) / prev_sales * 100
                return round(growth, 2)
            return 8.0
        except Exception:
            return 8.0
    
    def _predict_cpi(self, data: pd.DataFrame, month_offset: int) -> float:
        """预测CPI"""
        try:
            if data.empty or 'cpi' not in data.columns:
                return 102.0
            
            last_cpi = data['cpi'].iloc[-1]
            ma3 = data['cpi'].rolling(3).mean().iloc[-1]
            return round((last_cpi + ma3) / 2, 2)
        except Exception:
            return 102.0
    
    def _calculate_confidence(self, gdp_data: pd.DataFrame, economic_data: pd.DataFrame) -> float:
        """计算置信度"""
        try:
            if gdp_data.empty:
                return 0.5
            
            volatility = gdp_data['gdp_growth'].pct_change().std() if 'gdp_growth' in gdp_data.columns else 0.1
            confidence = max(0.3, min(0.95, 1 - volatility))
            return round(confidence, 2)
        except Exception:
            return 0.5
