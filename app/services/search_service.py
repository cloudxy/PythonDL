"""
全局搜索服务
"""
import json
from typing import List, Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, or_, and_, func
from sqlalchemy.orm import selectinload

from app.models.admin.user import User
from app.models.admin.role import Role
from app.models.admin.operation_log import OperationLog
from app.models.finance.stock import Stock
from app.models.weather.weather import Weather
from app.models.fortune.face_reading import FaceReading


class SearchService:
    """搜索服务类"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def global_search(
        self,
        keyword: str,
        types: Optional[List[str]] = None,
        page: int = 1,
        page_size: int = 20
    ) -> Dict[str, Any]:
        """
        全局搜索
        
        Args:
            keyword: 搜索关键词
            types: 搜索类型列表，如 ['users', 'stocks', 'weather']
            page: 页码
            page_size: 每页数量
        
        Returns:
            包含各类型搜索结果的字典
        """
        results = {
            "keyword": keyword,
            "total": 0,
            "results": {}
        }
        
        # 默认搜索所有类型
        if not types:
            types = ['users', 'stocks', 'weather', 'face_reading', 'operation_logs']
        
        # 并行搜索各类型
        search_tasks = []
        
        if 'users' in types:
            search_tasks.append(self._search_users(keyword, page, page_size))
        
        if 'stocks' in types:
            search_tasks.append(self._search_stocks(keyword, page, page_size))
        
        if 'weather' in types:
            search_tasks.append(self._search_weather(keyword, page, page_size))
        
        if 'face_analysis' in types:
            search_tasks.append(self._search_face_analysis(keyword, page, page_size))
        
        if 'operation_logs' in types:
            search_tasks.append(self._search_operation_logs(keyword, page, page_size))
        
        # 执行搜索
        search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
        
        # 整合结果
        result_names = ['users', 'stocks', 'weather', 'face_reading', 'operation_logs']
        for i, result_name in enumerate(result_names):
            if i < len(search_results) and not isinstance(search_results[i], Exception):
                results["results"][result_name] = search_results[i]
                results["total"] += search_results[i].get("count", 0)
        
        return results
    
    async def _search_users(
        self,
        keyword: str,
        page: int,
        page_size: int
    ) -> Dict[str, Any]:
        """搜索用户"""
        query = select(User).where(
            or_(
                User.username.ilike(f"%{keyword}%"),
                User.email.ilike(f"%{keyword}%"),
                User.real_name.ilike(f"%{keyword}%")
            )
        )
        
        # 总数
        count_query = select(func.count()).select_from(query.subquery())
        total = (await self.db.execute(count_query)).scalar()
        
        # 分页
        query = query.offset((page - 1) * page_size).limit(page_size)
        result = await self.db.execute(query)
        users = result.scalars().all()
        
        return {
            "count": total,
            "data": [
                {
                    "id": user.id,
                    "username": user.username,
                    "email": user.email,
                    "real_name": user.real_name,
                    "type": "user"
                }
                for user in users
            ]
        }
    
    async def _search_stocks(
        self,
        keyword: str,
        page: int,
        page_size: int
    ) -> Dict[str, Any]:
        """搜索股票"""
        query = select(Stock).where(
            or_(
                Stock.ts_code.ilike(f"%{keyword}%"),
                Stock.name.ilike(f"%{keyword}%") if hasattr(Stock, 'name') else False
            )
        )
        
        # 总数
        count_query = select(func.count()).select_from(query.subquery())
        total = (await self.db.execute(count_query)).scalar() or 0
        
        # 分页
        query = query.order_by(Stock.trade_date.desc())
        query = query.offset((page - 1) * page_size).limit(page_size)
        result = await self.db.execute(query)
        stocks = result.scalars().all()
        
        return {
            "count": total,
            "data": [
                {
                    "id": stock.id,
                    "ts_code": stock.ts_code,
                    "trade_date": stock.trade_date.isoformat() if stock.trade_date else None,
                    "close": float(stock.close) if stock.close else 0,
                    "type": "stock"
                }
                for stock in stocks
            ]
        }
    
    async def _search_weather(
        self,
        keyword: str,
        page: int,
        page_size: int
    ) -> Dict[str, Any]:
        """搜索气象数据"""
        query = select(Weather).where(
            or_(
                Weather.station_id.ilike(f"%{keyword}%"),
            )
        )
        
        # 总数
        count_query = select(func.count()).select_from(query.subquery())
        total = (await self.db.execute(count_query)).scalar() or 0
        
        # 分页
        query = query.order_by(Weather.record_date.desc())
        query = query.offset((page - 1) * page_size).limit(page_size)
        result = await self.db.execute(query)
        records = result.scalars().all()
        
        return {
            "count": total,
            "data": [
                {
                    "id": record.id,
                    "station_id": record.station_id,
                    "date": record.record_date.isoformat() if record.record_date else None,
                    "type": "weather"
                }
                for record in records
            ]
        }
    
    async def _search_face_analysis(
        self,
        keyword: str,
        page: int,
        page_size: int
    ) -> Dict[str, Any]:
        """搜索面相分析记录"""
        query = select(FaceReading).where(
            or_(
                FaceReading.name.ilike(f"%{keyword}%"),
                FaceReading.meaning.ilike(f"%{keyword}%"),
                FaceReading.interpretation.ilike(f"%{keyword}%")
            )
        )
        
        # 总数
        count_query = select(func.count()).select_from(query.subquery())
        total = (await self.db.execute(count_query)).scalar() or 0
        
        # 分页
        query = query.order_by(FaceReading.created_at.desc())
        query = query.offset((page - 1) * page_size).limit(page_size)
        result = await self.db.execute(query)
        records = result.scalars().all()
        
        return {
            "count": total,
            "data": [
                {
                    "id": record.id,
                    "name": record.name,
                    "face_part": record.face_part,
                    "interpretation": record.interpretation[:100] + "..." if record.interpretation and len(record.interpretation) > 100 else record.interpretation,
                    "created_at": record.created_at.isoformat() if record.created_at else None,
                    "type": "face_reading"
                }
                for record in records
            ]
        }
    
    async def _search_operation_logs(
        self,
        keyword: str,
        page: int,
        page_size: int
    ) -> Dict[str, Any]:
        """搜索操作日志"""
        query = select(OperationLog).where(
            or_(
                OperationLog.module.ilike(f"%{keyword}%"),
                OperationLog.action.ilike(f"%{keyword}%"),
                OperationLog.ip_address.ilike(f"%{keyword}%")
            )
        )
        
        # 总数
        count_query = select(func.count()).select_from(query.subquery())
        total = (await self.db.execute(count_query)).scalar() or 0
        
        # 分页
        query = query.order_by(OperationLog.created_at.desc())
        query = query.offset((page - 1) * page_size).limit(page_size)
        result = await self.db.execute(query)
        logs = result.scalars().all()
        
        return {
            "count": total,
            "data": [
                {
                    "id": log.id,
                    "module": log.module,
                    "action": log.action,
                    "ip_address": log.ip_address,
                    "created_at": log.created_at.isoformat() if log.created_at else None,
                    "type": "operation_log"
                }
                for log in logs
            ]
        }
    
    async def advanced_search(
        self,
        filters: Dict[str, Any],
        model_type: str,
        page: int = 1,
        page_size: int = 20
    ) -> Dict[str, Any]:
        """
        高级搜索
        
        Args:
            filters: 过滤条件字典
            model_type: 模型类型
            page: 页码
            page_size: 每页数量
        
        Returns:
            搜索结果
        """
        # 根据模型类型构建查询
        if model_type == "stock":
            return await self._advanced_search_stock(filters, page, page_size)
        elif model_type == "weather":
            return await self._advanced_search_weather(filters, page, page_size)
        else:
            return {"count": 0, "data": []}
    
    async def _advanced_search_stock(
        self,
        filters: Dict[str, Any],
        page: int,
        page_size: int
    ) -> Dict[str, Any]:
        """高级搜索股票"""
        conditions = []
        
        # 构建动态查询条件
        if 'start_date' in filters and 'end_date' in filters:
            conditions.append(
                and_(
                    Stock.trade_date >= filters['start_date'],
                    Stock.trade_date <= filters['end_date']
                )
            )
        
        if 'min_price' in filters:
            conditions.append(Stock.close >= filters['min_price'])
        
        if 'max_price' in filters:
            conditions.append(Stock.close <= filters['max_price'])
        
        if 'ts_code' in filters:
            conditions.append(Stock.ts_code.ilike(f"%{filters['ts_code']}%"))
        
        query = select(Stock).where(and_(*conditions)) if conditions else select(Stock)
        
        # 总数
        count_query = select(func.count()).select_from(query.subquery())
        total = (await self.db.execute(count_query)).scalar() or 0
        
        # 分页
        query = query.order_by(Stock.trade_date.desc())
        query = query.offset((page - 1) * page_size).limit(page_size)
        result = await self.db.execute(query)
        stocks = result.scalars().all()
        
        return {
            "count": total,
            "data": [
                {
                    "id": stock.id,
                    "ts_code": stock.ts_code,
                    "trade_date": stock.trade_date.isoformat() if stock.trade_date else None,
                    "close": float(stock.close) if stock.close else 0,
                    "change_pct": float(stock.change_pct) if hasattr(stock, 'change_pct') and stock.change_pct else 0
                }
                for stock in stocks
            ]
        }
    
    async def _advanced_search_weather(
        self,
        filters: Dict[str, Any],
        page: int,
        page_size: int
    ) -> Dict[str, Any]:
        """高级搜索气象数据"""
        conditions = []
        
        if 'start_date' in filters and 'end_date' in filters:
            conditions.append(
                and_(
                    Weather.record_date >= filters['start_date'],
                    Weather.record_date <= filters['end_date']
                )
            )
        
        if 'station_id' in filters:
            conditions.append(Weather.station_id.ilike(f"%{filters['station_id']}%"))
        
        if 'min_temp' in filters:
            conditions.append(Weather.avg_temp >= filters['min_temp'])
        
        if 'max_temp' in filters:
            conditions.append(Weather.avg_temp <= filters['max_temp'])
        
        query = select(Weather).where(and_(*conditions)) if conditions else select(Weather)
        
        # 总数
        count_query = select(func.count()).select_from(query.subquery())
        total = (await self.db.execute(count_query)).scalar() or 0
        
        # 分页
        query = query.order_by(Weather.record_date.desc())
        query = query.offset((page - 1) * page_size).limit(page_size)
        result = await self.db.execute(query)
        records = result.scalars().all()
        
        return {
            "count": total,
            "data": [
                {
                    "id": record.id,
                    "station_id": record.station_id,
                    "date": record.record_date.isoformat() if record.record_date else None,
                    "avg_temp": float(record.avg_temp) if record.avg_temp else None,
                    "precipitation": float(record.precipitation) if record.precipitation else None
                }
                for record in records
            ]
        }
