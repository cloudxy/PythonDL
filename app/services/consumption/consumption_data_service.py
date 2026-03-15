"""消费数据服务层

此模块提供消费数据（GDP、人口、经济指标）的数据库写入和管理功能。
"""
import asyncio
import logging
from datetime import datetime
from decimal import Decimal
from typing import List, Dict, Optional, Any
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError

from app.models.consumption.gdp_data import GDPData as GDPDataModel
from app.models.consumption.population_data import PopulationData
from app.models.consumption.economic_indicator import EconomicIndicator
from app.core.logger import get_logger

logger = get_logger("consumption_data_service")


class ConsumptionDataService:
    """消费数据服务"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def save_gdp_data(self, gdp_data: Dict[str, Any]) -> Optional[GDPDataModel]:
        """保存 GDP 数据"""
        try:
            region_code = gdp_data.get("region_code", "")
            year = gdp_data.get("year", 0)
            
            # 检查是否已存在
            stmt = select(GDPDataModel).where(
                GDPDataModel.region_code == region_code,
                GDPDataModel.year == year
            )
            result = await self.db.execute(stmt)
            existing = result.scalar_one_or_none()
            
            if existing:
                # 更新
                for key, value in gdp_data.items():
                    if hasattr(existing, key):
                        setattr(existing, key, value)
                existing.updated_at = datetime.utcnow()
            else:
                # 新增
                gdp = GDPDataModel(**gdp_data)
                self.db.add(gdp)
                existing = gdp
            
            await self.db.commit()
            await self.db.refresh(existing)
            return existing
            
        except SQLAlchemyError as e:
            await self.db.rollback()
            logger.error(f"保存 GDP 数据失败：{e}")
            return None
        except Exception as e:
            await self.db.rollback()
            logger.error(f"保存 GDP 数据异常：{e}")
            return None
    
    async def save_population_data(self, pop_data: Dict[str, Any]) -> Optional[PopulationData]:
        """保存人口数据"""
        try:
            region_code = pop_data.get("region_code", "")
            year = pop_data.get("year", 0)
            
            stmt = select(PopulationData).where(
                PopulationData.region_code == region_code,
                PopulationData.year == year
            )
            result = await self.db.execute(stmt)
            existing = result.scalar_one_or_none()
            
            if existing:
                for key, value in pop_data.items():
                    if hasattr(existing, key):
                        setattr(existing, key, value)
                existing.updated_at = datetime.utcnow()
            else:
                population = PopulationData(**pop_data)
                self.db.add(population)
                existing = population
            
            await self.db.commit()
            await self.db.refresh(existing)
            return existing
            
        except SQLAlchemyError as e:
            await self.db.rollback()
            logger.error(f"保存人口数据失败：{e}")
            return None
        except Exception as e:
            await self.db.rollback()
            logger.error(f"保存人口数据异常：{e}")
            return None
    
    async def save_economic_indicator(self, indicator_data: Dict[str, Any]) -> Optional[EconomicIndicator]:
        """保存经济指标数据"""
        try:
            region_code = indicator_data.get("region_code", "")
            year = indicator_data.get("year", 0)
            month = indicator_data.get("month", 0)
            
            stmt = select(EconomicIndicator).where(
                EconomicIndicator.region_code == region_code,
                EconomicIndicator.year == year,
                EconomicIndicator.month == month
            )
            result = await self.db.execute(stmt)
            existing = result.scalar_one_or_none()
            
            if existing:
                for key, value in indicator_data.items():
                    if hasattr(existing, key):
                        setattr(existing, key, value)
                existing.updated_at = datetime.utcnow()
            else:
                indicator = EconomicIndicator(**indicator_data)
                self.db.add(indicator)
                existing = indicator
            
            await self.db.commit()
            await self.db.refresh(existing)
            return existing
            
        except SQLAlchemyError as e:
            await self.db.rollback()
            logger.error(f"保存经济指标数据失败：{e}")
            return None
        except Exception as e:
            await self.db.rollback()
            logger.error(f"保存经济指标数据异常：{e}")
            return None
    
    async def save_batch(
        self,
        gdp_data_list: Optional[List[Dict[str, Any]]] = None,
        population_data_list: Optional[List[Dict[str, Any]]] = None,
        economic_indicator_list: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, int]:
        """批量保存消费数据"""
        results = {
            "gdp_saved": 0,
            "population_saved": 0,
            "economic_indicator_saved": 0
        }
        
        if gdp_data_list:
            for data in gdp_data_list:
                if await self.save_gdp_data(data):
                    results["gdp_saved"] += 1
        
        if population_data_list:
            for data in population_data_list:
                if await self.save_population_data(data):
                    results["population_saved"] += 1
        
        if economic_indicator_list:
            for data in economic_indicator_list:
                if await self.save_economic_indicator(data):
                    results["economic_indicator_saved"] += 1
        
        logger.info(f"批量保存消费数据：GDP={results['gdp_saved']}, "
                   f"人口={results['population_saved']}, "
                   f"经济指标={results['economic_indicator_saved']}")
        
        return results


async def save_consumption_to_db(
    db: AsyncSession,
    data_type: str,
    data: Dict[str, Any]
) -> bool:
    """保存消费数据到数据库的便捷函数"""
    service = ConsumptionDataService(db)
    
    if data_type == "gdp":
        result = await service.save_gdp_data(data)
    elif data_type == "population":
        result = await service.save_population_data(data)
    elif data_type == "economic_indicator":
        result = await service.save_economic_indicator(data)
    else:
        logger.warning(f"未知的数据类型：{data_type}")
        return False
    
    return result is not None


async def save_consumption_batch_to_db(
    db: AsyncSession,
    gdp_data: Optional[List[Dict[str, Any]]] = None,
    population_data: Optional[List[Dict[str, Any]]] = None,
    economic_indicator_data: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, int]:
    """批量保存消费数据到数据库"""
    service = ConsumptionDataService(db)
    return await service.save_batch(gdp_data, population_data, economic_indicator_data)
