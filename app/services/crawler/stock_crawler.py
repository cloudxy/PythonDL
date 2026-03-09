"""股票数据爬虫

此模块提供股票数据采集功能。
"""
import requests
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, date, timedelta
from sqlalchemy.orm import Session
import time
import json

from app.models.finance.stock_basic import StockBasic
from app.models.finance.stock import Stock
from app.core.logger import get_logger

logger = get_logger("stock_crawler")


class StockCrawler:
    """股票数据爬虫类"""
    
    def __init__(self, db: Session):
        self.db = db
        self.status = {
            "is_running": False,
            "last_run": None,
            "total_records": 0,
            "error_count": 0
        }
    
    def crawl_stock_data(self, days: int = 30) -> Dict[str, Any]:
        """采集股票数据"""
        self.status["is_running"] = True
        start_time = datetime.now()
        
        try:
            logger.info(f"开始采集股票数据，天数: {days}")
            
            self._crawl_stock_basics()
            
            total_records = self._crawl_stock_quotes(days)
            
            self.status["is_running"] = False
            self.status["last_run"] = datetime.now()
            self.status["total_records"] = total_records
            
            logger.info(f"股票数据采集完成，共采集 {total_records} 条记录")
            
            return {
                "success": True,
                "total_records": total_records,
                "duration": (datetime.now() - start_time).seconds
            }
            
        except Exception as e:
            self.status["is_running"] = False
            self.status["error_count"] += 1
            logger.error(f"股票数据采集失败: {str(e)}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _crawl_stock_basics(self) -> int:
        """采集股票基础信息"""
        logger.info("开始采集股票基础信息")
        
        try:
            mock_data = self._generate_mock_stock_basics()
            
            count = 0
            for data in mock_data:
                existing = self.db.query(StockBasic).filter(
                    StockBasic.ts_code == data["ts_code"]
                ).first()
                
                if not existing:
                    stock = StockBasic(**data)
                    self.db.add(stock)
                    count += 1
            
            self.db.commit()
            logger.info(f"股票基础信息采集完成，新增 {count} 条记录")
            return count
            
        except Exception as e:
            logger.error(f"采集股票基础信息失败: {str(e)}")
            return 0
    
    def _crawl_stock_quotes(self, days: int) -> int:
        """采集股票行情数据"""
        logger.info(f"开始采集股票行情数据，天数: {days}")
        
        try:
            stocks = self.db.query(StockBasic).limit(100).all()
            total_count = 0
            
            for stock in stocks:
                mock_data = self._generate_mock_stock_quotes(stock.ts_code, days)
                
                for data in mock_data:
                    existing = self.db.query(Stock).filter(
                        Stock.ts_code == data["ts_code"],
                        Stock.trade_date == data["trade_date"]
                    ).first()
                    
                    if not existing:
                        stock_quote = Stock(**data)
                        self.db.add(stock_quote)
                        total_count += 1
                
                if total_count % 100 == 0:
                    self.db.commit()
                    logger.info(f"已采集 {total_count} 条行情数据")
            
            self.db.commit()
            logger.info(f"股票行情数据采集完成，新增 {total_count} 条记录")
            return total_count
            
        except Exception as e:
            logger.error(f"采集股票行情数据失败: {str(e)}")
            return 0
    
    def _generate_mock_stock_basics(self) -> List[Dict[str, Any]]:
        """生成模拟股票基础数据"""
        stocks = []
        prefixes = ["60", "00", "30", "68"]
        
        for i in range(100):
            prefix = prefixes[i % len(prefixes)]
            code = f"{prefix}{str(i).zfill(4)}"
            ts_code = f"{code}.SH" if prefix in ["60", "68"] else f"{code}.SZ"
            
            stocks.append({
                "ts_code": ts_code,
                "symbol": code,
                "name": f"测试股票{i+1}",
                "area": "北京",
                "industry": "科技",
                "market": "主板" if prefix in ["60", "00"] else "创业板",
                "exchange": "SSE" if prefix in ["60", "68"] else "SZSE",
                "list_status": "L",
                "list_date": date(2020, 1, 1) + timedelta(days=i)
            })
        
        return stocks
    
    def _generate_mock_stock_quotes(
        self,
        ts_code: str,
        days: int
    ) -> List[Dict[str, Any]]:
        """生成模拟股票行情数据"""
        import random
        
        quotes = []
        base_price = random.uniform(10, 100)
        
        for i in range(days):
            trade_date = date.today() - timedelta(days=days-i)
            
            open_price = base_price * (1 + random.uniform(-0.05, 0.05))
            close_price = open_price * (1 + random.uniform(-0.03, 0.03))
            high_price = max(open_price, close_price) * (1 + random.uniform(0, 0.02))
            low_price = min(open_price, close_price) * (1 - random.uniform(0, 0.02))
            
            change = close_price - base_price
            pct_chg = (change / base_price) * 100
            
            quotes.append({
                "ts_code": ts_code,
                "trade_date": trade_date,
                "open": round(open_price, 2),
                "high": round(high_price, 2),
                "low": round(low_price, 2),
                "close": round(close_price, 2),
                "pre_close": round(base_price, 2),
                "change": round(change, 2),
                "pct_chg": round(pct_chg, 2),
                "vol": random.randint(100000, 10000000),
                "amount": round(random.uniform(1000000, 100000000), 2)
            })
            
            base_price = close_price
        
        return quotes
    
    def get_status(self) -> Dict[str, Any]:
        """获取爬虫状态"""
        return self.status
