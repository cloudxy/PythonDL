"""股票数据服务层

此模块提供股票数据的数据库写入和管理功能。
"""
import asyncio
import logging
from datetime import datetime, date
from decimal import Decimal
from typing import List, Dict, Optional, Any
from sqlalchemy import select, update, delete
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.exc import SQLAlchemyError

from app.models.finance.stock import Stock
from app.models.finance.stock_basic import StockBasic
from app.core.logger import get_logger

logger = get_logger("stock_data_service")


class StockDataService:
    """股票数据服务"""
    
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def save_stock_basic(self, stock_data: Dict[str, Any]) -> Optional[StockBasic]:
        """保存股票基础信息"""
        try:
            ts_code = stock_data.get("ts_code", "")
            
            # 检查是否已存在
            stmt = select(StockBasic).where(StockBasic.ts_code == ts_code)
            result = await self.db.execute(stmt)
            existing = result.scalar_one_or_none()
            
            if existing:
                # 更新
                for key, value in stock_data.items():
                    if hasattr(existing, key):
                        setattr(existing, key, value)
                existing.updated_at = datetime.utcnow()
                logger.debug(f"更新股票基础信息：{ts_code}")
            else:
                # 新增
                stock_basic = StockBasic(**stock_data)
                self.db.add(stock_basic)
                existing = stock_basic
                logger.debug(f"新增股票基础信息：{ts_code}")
            
            await self.db.commit()
            await self.db.refresh(existing if existing else stock_basic)
            return existing if existing else stock_basic
            
        except SQLAlchemyError as e:
            await self.db.rollback()
            logger.error(f"保存股票基础信息失败：{e}")
            return None
        except Exception as e:
            await self.db.rollback()
            logger.error(f"保存股票基础信息异常：{e}")
            return None
    
    async def save_stock_quote(self, quote_data: Dict[str, Any]) -> Optional[Stock]:
        """保存股票行情数据"""
        try:
            ts_code = quote_data.get("ts_code", "")
            trade_date = quote_data.get("trade_date", None)
            
            if not ts_code or not trade_date:
                logger.warning("缺少必要字段：ts_code 或 trade_date")
                return None
            
            # 检查是否已存在
            stmt = select(Stock).where(
                Stock.ts_code == ts_code,
                Stock.trade_date == trade_date
            )
            result = await self.db.execute(stmt)
            existing = result.scalar_one_or_none()
            
            if existing:
                # 更新
                for key, value in quote_data.items():
                    if hasattr(existing, key):
                        setattr(existing, key, value)
                existing.updated_at = datetime.utcnow()
                logger.debug(f"更新股票行情：{ts_code} - {trade_date}")
            else:
                # 新增
                stock = Stock(**quote_data)
                self.db.add(stock)
                existing = stock
                logger.debug(f"新增股票行情：{ts_code} - {trade_date}")
            
            await self.db.commit()
            await self.db.refresh(existing if existing else stock)
            return existing if existing else stock
            
        except SQLAlchemyError as e:
            await self.db.rollback()
            logger.error(f"保存股票行情失败：{e}")
            return None
        except Exception as e:
            await self.db.rollback()
            logger.error(f"保存股票行情异常：{e}")
            return None
    
    async def save_stock_quotes_batch(
        self,
        quotes: List[Dict[str, Any]]
    ) -> int:
        """批量保存股票行情数据"""
        if not quotes:
            return 0
        
        success_count = 0
        for quote in quotes:
            result = await self.save_stock_quote(quote)
            if result:
                success_count += 1
        
        logger.info(f"批量保存股票行情：成功{success_count}/{len(quotes)}条")
        return success_count
    
    async def get_stock_by_code(self, ts_code: str) -> Optional[StockBasic]:
        """获取股票基础信息"""
        try:
            stmt = select(StockBasic).where(StockBasic.ts_code == ts_code)
            result = await self.db.execute(stmt)
            return result.scalar_one_or_none()
        except Exception as e:
            logger.error(f"获取股票基础信息失败：{e}")
            return None
    
    async def get_stock_quotes(
        self,
        ts_code: str,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        limit: int = 100
    ) -> List[Stock]:
        """获取股票行情数据"""
        try:
            stmt = select(Stock).where(Stock.ts_code == ts_code)
            
            if start_date:
                stmt = stmt.where(Stock.trade_date >= start_date)
            if end_date:
                stmt = stmt.where(Stock.trade_date <= end_date)
            
            stmt = stmt.order_by(Stock.trade_date.desc()).limit(limit)
            
            result = await self.db.execute(stmt)
            return result.scalars().all()
        except Exception as e:
            logger.error(f"获取股票行情数据失败：{e}")
            return []
    
    async def delete_old_data(self, days: int = 365) -> int:
        """删除旧数据"""
        try:
            from datetime import timedelta
            cutoff_date = date.today() - timedelta(days=days)
            
            stmt = delete(Stock).where(Stock.trade_date < cutoff_date)
            result = await self.db.execute(stmt)
            await self.db.commit()
            
            logger.info(f"删除{days}天前的股票数据：{result.rowcount}条")
            return result.rowcount
        except Exception as e:
            await self.db.rollback()
            logger.error(f"删除旧数据失败：{e}")
            return 0


async def save_stock_to_db(
    db: AsyncSession,
    stock_data: Dict[str, Any],
    quote_data: Optional[Dict[str, Any]] = None
) -> bool:
    """保存股票数据到数据库的便捷函数"""
    service = StockDataService(db)
    
    # 保存基础信息
    if stock_data:
        result = await service.save_stock_basic(stock_data)
        if not result:
            return False
    
    # 保存行情数据
    if quote_data:
        result = await service.save_stock_quote(quote_data)
        if not result:
            return False
    
    return True


async def save_stocks_batch_to_db(
    db: AsyncSession,
    stocks: List[Dict[str, Any]],
    quotes: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, int]:
    """批量保存股票数据到数据库"""
    service = StockDataService(db)
    
    results = {
        "stocks_saved": 0,
        "quotes_saved": 0
    }
    
    # 保存股票基础信息
    for stock in stocks:
        result = await service.save_stock_basic(stock)
        if result:
            results["stocks_saved"] += 1
    
    # 保存行情数据
    if quotes:
        results["quotes_saved"] = await service.save_stock_quotes_batch(quotes)
    
    return results
