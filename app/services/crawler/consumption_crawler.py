"""宏观消费数据爬虫

此模块提供宏观消费数据采集功能。
"""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, date
from sqlalchemy.orm import Session
import random

from app.models.consumption.gdp_data import GDPData
from app.models.consumption.population_data import PopulationData
from app.models.consumption.economic_indicator import EconomicIndicator
from app.models.consumption.community_data import CommunityData
from app.core.logger import get_logger

logger = get_logger("consumption_crawler")


class ConsumptionCrawler:
    """宏观消费数据爬虫类"""
    
    def __init__(self, db: Session):
        self.db = db
        self.status = {
            "is_running": False,
            "last_run": None,
            "total_records": 0,
            "error_count": 0
        }
    
    def crawl_consumption_data(
        self,
        data_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """采集宏观消费数据"""
        self.status["is_running"] = True
        start_time = datetime.now()
        
        if data_types is None:
            data_types = ["gdp", "population", "economic_indicator", "community"]
        
        try:
            logger.info(f"开始采集宏观消费数据，类型: {data_types}")
            
            total_records = 0
            
            if "gdp" in data_types:
                total_records += self._crawl_gdp_data()
            
            if "population" in data_types:
                total_records += self._crawl_population_data()
            
            if "economic_indicator" in data_types:
                total_records += self._crawl_economic_indicator()
            
            if "community" in data_types:
                total_records += self._crawl_community_data()
            
            self.status["is_running"] = False
            self.status["last_run"] = datetime.now()
            self.status["total_records"] = total_records
            
            logger.info(f"宏观消费数据采集完成，共采集 {total_records} 条记录")
            
            return {
                "success": True,
                "total_records": total_records,
                "duration": (datetime.now() - start_time).seconds
            }
            
        except Exception as e:
            self.status["is_running"] = False
            self.status["error_count"] += 1
            logger.error(f"宏观消费数据采集失败: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _crawl_gdp_data(self) -> int:
        """采集GDP数据"""
        logger.info("开始采集GDP数据")
        
        try:
            data_list = self._generate_gdp_data()
            
            count = 0
            for data in data_list:
                existing = self.db.query(GDPData).filter(
                    GDPData.region_code == data["region_code"],
                    GDPData.year == data["year"]
                ).first()
                
                if not existing:
                    item = GDPData(**data)
                    self.db.add(item)
                    count += 1
            
            self.db.commit()
            logger.info(f"GDP数据采集完成，新增 {count} 条记录")
            return count
            
        except Exception as e:
            logger.error(f"采集GDP数据失败: {str(e)}")
            return 0
    
    def _crawl_population_data(self) -> int:
        """采集人口数据"""
        logger.info("开始采集人口数据")
        
        try:
            data_list = self._generate_population_data()
            
            count = 0
            for data in data_list:
                existing = self.db.query(PopulationData).filter(
                    PopulationData.region_code == data["region_code"],
                    PopulationData.year == data["year"]
                ).first()
                
                if not existing:
                    item = PopulationData(**data)
                    self.db.add(item)
                    count += 1
            
            self.db.commit()
            logger.info(f"人口数据采集完成，新增 {count} 条记录")
            return count
            
        except Exception as e:
            logger.error(f"采集人口数据失败: {str(e)}")
            return 0
    
    def _crawl_economic_indicator(self) -> int:
        """采集经济指标数据"""
        logger.info("开始采集经济指标数据")
        
        try:
            data_list = self._generate_economic_indicator_data()
            
            count = 0
            for data in data_list:
                existing = self.db.query(EconomicIndicator).filter(
                    EconomicIndicator.region_code == data["region_code"],
                    EconomicIndicator.year == data["year"],
                    EconomicIndicator.month == data["month"]
                ).first()
                
                if not existing:
                    item = EconomicIndicator(**data)
                    self.db.add(item)
                    count += 1
            
            self.db.commit()
            logger.info(f"经济指标数据采集完成，新增 {count} 条记录")
            return count
            
        except Exception as e:
            logger.error(f"采集经济指标数据失败: {str(e)}")
            return 0
    
    def _crawl_community_data(self) -> int:
        """采集小区数据"""
        logger.info("开始采集小区数据")
        return 0
    
    def _generate_gdp_data(self) -> List[Dict[str, Any]]:
        """生成GDP数据"""
        from decimal import Decimal
        
        regions = [
            {"code": "CN", "name": "中国", "level": "国家"},
            {"code": "BJ", "name": "北京", "level": "省"},
            {"code": "SH", "name": "上海", "level": "省"},
            {"code": "GD", "name": "广东", "level": "省"},
        ]
        
        data_list = []
        
        for year in range(2020, 2025):
            for region in regions:
                base_gdp = random.randint(100000, 10000000)
                data_list.append({
                    "region_code": region["code"],
                    "region_name": region["name"],
                    "region_level": region["level"],
                    "year": year,
                    "gdp": base_gdp * 10000,
                    "gdp_growth": round(Decimal(random.uniform(5, 10)), 2),
                    "primary_industry": int(base_gdp * 0.1 * 10000),
                    "secondary_industry": int(base_gdp * 0.4 * 10000),
                    "tertiary_industry": int(base_gdp * 0.5 * 10000),
                    "per_capita_gdp": round(Decimal(random.uniform(50000, 150000)), 2)
                })
        
        return data_list
    
    def _generate_population_data(self) -> List[Dict[str, Any]]:
        """生成人口数据"""
        from decimal import Decimal
        
        regions = [
            {"code": "CN", "name": "中国", "level": "国家"},
            {"code": "BJ", "name": "北京", "level": "省"},
            {"code": "SH", "name": "上海", "level": "省"},
        ]
        
        data_list = []
        
        for year in range(2020, 2025):
            for region in regions:
                total = random.randint(10000000, 1400000000)
                data_list.append({
                    "region_code": region["code"],
                    "region_name": region["name"],
                    "region_level": region["level"],
                    "year": year,
                    "total_population": total,
                    "male_population": int(total * 0.51),
                    "female_population": int(total * 0.49),
                    "urban_population": int(total * 0.6),
                    "rural_population": int(total * 0.4),
                    "urbanization_rate": round(Decimal(60.0), 2),
                    "birth_rate": round(Decimal(random.uniform(8, 12)), 2),
                    "death_rate": round(Decimal(random.uniform(6, 8)), 2)
                })
        
        return data_list
    
    def _generate_economic_indicator_data(self) -> List[Dict[str, Any]]:
        """生成经济指标数据"""
        from decimal import Decimal
        
        data_list = []
        
        for year in range(2023, 2025):
            for month in range(1, 13):
                if year == 2025 and month > 3:
                    continue
                
                data_list.append({
                    "region_code": "CN",
                    "region_name": "中国",
                    "region_level": "国家",
                    "year": year,
                    "month": month,
                    "cpi": round(Decimal(random.uniform(100, 105)), 2),
                    "cpi_yoy": round(Decimal(random.uniform(1, 3)), 2),
                    "ppi": round(Decimal(random.uniform(95, 105)), 2),
                    "ppi_yoy": round(Decimal(random.uniform(-2, 2)), 2),
                    "pmi": round(Decimal(random.uniform(48, 52)), 2),
                    "retail_sales": random.randint(300000000, 500000000),
                    "retail_sales_yoy": round(Decimal(random.uniform(5, 10)), 2)
                })
        
        return data_list
    
    def get_status(self) -> Dict[str, Any]:
        """获取爬虫状态"""
        return self.status
