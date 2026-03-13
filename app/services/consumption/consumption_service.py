from typing import List, Optional, Tuple
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, and_, func
from app.models.consumption.gdp_data import GDPData
from app.models.consumption.population_data import PopulationData
from app.models.consumption.economic_indicator import EconomicIndicator
from app.models.consumption.community_data import CommunityData
from datetime import datetime


class ConsumptionService:
    """消费分析服务类（统一入口）"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
        self.gdp = GDPService(db)
        self.population = PopulationService(db)
        self.economic = EconomicIndicatorService(db)
        self.community = CommunityService(db)


class GDPService:
    """GDP 数据服务"""
    
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_gdp_data(self, data: dict) -> GDPData:
        """创建 GDP 数据"""
        try:
            gdp = GDPData(
                region=data.get("region"),
                year=data.get("year"),
                quarter=data.get("quarter"),
                gdp_value=data.get("gdp_value"),
                gdp_growth=data.get("gdp_growth"),
                primary_industry=data.get("primary_industry"),
                secondary_industry=data.get("secondary_industry"),
                tertiary_industry=data.get("tertiary_industry"),
                created_at=datetime.now()
            )
            self.db.add(gdp)
            await self.db.commit()
            await self.db.refresh(gdp)
            return gdp
        except Exception:
            await self.db.rollback()
            raise

    async def get_gdp_data(self, gdp_id: int) -> Optional[GDPData]:
        """获取 GDP 数据"""
        result = await self.db.execute(select(GDPData).where(GDPData.id == gdp_id))
        return result.scalar_one_or_none()

    async def get_gdp_by_region(
        self, 
        region: str, 
        years: int = 5
    ) -> List[GDPData]:
        """按地区获取 GDP 数据"""
        current_year = datetime.now().year
        start_year = current_year - years
        
        result = await self.db.execute(
            select(GDPData)
            .where(
                and_(
                    GDPData.region == region,
                    GDPData.year >= start_year
                )
            )
            .order_by(GDPData.year.desc(), GDPData.quarter.desc())
        )
        return result.scalars().all()


class PopulationService:
    """人口数据服务"""
    
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_population_data(self, data: dict) -> PopulationData:
        """创建人口数据"""
        try:
            population = PopulationData(
                region=data.get("region"),
                year=data.get("year"),
                total_population=data.get("total_population"),
                urban_population=data.get("urban_population"),
                rural_population=data.get("rural_population"),
                birth_rate=data.get("birth_rate"),
                death_rate=data.get("death_rate"),
                natural_growth=data.get("natural_growth"),
                created_at=datetime.now()
            )
            self.db.add(population)
            await self.db.commit()
            await self.db.refresh(population)
            return population
        except Exception:
            await self.db.rollback()
            raise

    async def get_population_data(self, population_id: int) -> Optional[PopulationData]:
        """获取人口数据"""
        result = await self.db.execute(
            select(PopulationData).where(PopulationData.id == population_id)
        )
        return result.scalar_one_or_none()

    async def get_population_by_region(
        self, 
        region: str, 
        years: int = 10
    ) -> List[PopulationData]:
        """按地区获取人口数据"""
        current_year = datetime.now().year
        start_year = current_year - years
        
        result = await self.db.execute(
            select(PopulationData)
            .where(
                and_(
                    PopulationData.region == region,
                    PopulationData.year >= start_year
                )
            )
            .order_by(PopulationData.year.desc())
        )
        return result.scalars().all()


class EconomicIndicatorService:
    """经济指标服务"""
    
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_economic_indicator(self, data: dict) -> EconomicIndicator:
        """创建经济指标数据"""
        try:
            economic = EconomicIndicator(
                region=data.get("region"),
                year=data.get("year"),
                month=data.get("month"),
                cpi=data.get("cpi"),
                ppi=data.get("ppi"),
                unemployment_rate=data.get("unemployment_rate"),
                retail_sales=data.get("retail_sales"),
                import_export=data.get("import_export"),
                created_at=datetime.now()
            )
            self.db.add(economic)
            await self.db.commit()
            await self.db.refresh(economic)
            return economic
        except Exception:
            await self.db.rollback()
            raise

    async def get_economic_indicator(self, indicator_id: int) -> Optional[EconomicIndicator]:
        """获取经济指标数据"""
        result = await self.db.execute(
            select(EconomicIndicator).where(EconomicIndicator.id == indicator_id)
        )
        return result.scalar_one_or_none()

    async def get_economic_by_region(
        self, 
        region: str, 
        years: int = 3
    ) -> List[EconomicIndicator]:
        """按地区获取经济指标数据"""
        current_year = datetime.now().year
        start_year = current_year - years
        
        result = await self.db.execute(
            select(EconomicIndicator)
            .where(
                and_(
                    EconomicIndicator.region == region,
                    EconomicIndicator.year >= start_year
                )
            )
            .order_by(EconomicIndicator.year.desc(), EconomicIndicator.month.desc())
        )
        return result.scalars().all()


class CommunityService:
    """小区数据服务"""
    
    def __init__(self, db: AsyncSession):
        self.db = db

    async def create_community_data(self, data: dict) -> CommunityData:
        """创建小区数据"""
        try:
            community = CommunityData(
                city=data.get("city"),
                district=data.get("district"),
                community_name=data.get("community_name"),
                total_households=data.get("total_households"),
                total_population=data.get("total_population"),
                avg_price=data.get("avg_price"),
                avg_area=data.get("avg_area"),
                facility_score=data.get("facility_score"),
                created_at=datetime.now()
            )
            self.db.add(community)
            await self.db.commit()
            await self.db.refresh(community)
            return community
        except Exception:
            await self.db.rollback()
            raise

    async def get_community_data(self, community_id: int) -> Optional[CommunityData]:
        """获取小区数据"""
        result = await self.db.execute(
            select(CommunityData).where(CommunityData.id == community_id)
        )
        return result.scalar_one_or_none()

    async def get_communities_by_city(
        self, 
        city: str, 
        skip: int = 0, 
        limit: int = 100
    ) -> List[CommunityData]:
        """按城市获取小区数据"""
        result = await self.db.execute(
            select(CommunityData)
            .where(CommunityData.city == city)
            .order_by(CommunityData.facility_score.desc())
            .offset(skip)
            .limit(limit)
        )
        return result.scalars().all()
