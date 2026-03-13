from typing import List, Optional, Dict, Any
from sqlalchemy.orm import Session
from app.models.fortune import FaceReading, BaziReading, ZhouyiReading, Constellation, DailyFortune
from datetime import datetime


class FaceReadingService:
    def __init__(self, db: Session):
        self.db = db

    def create_face_reading(self, data: dict) -> FaceReading:
        try:
            reading = FaceReading(
                user_id=data.get("user_id"),
                face_shape=data.get("face_shape"),
                forehead=data.get("forehead"),
                eyes=data.get("eyes"),
                nose=data.get("nose"),
                mouth=data.get("mouth"),
                chin=data.get("chin"),
                overall_score=data.get("overall_score"),
                analysis=data.get("analysis"),
                created_at=datetime.now()
            )
            self.db.add(reading)
            self.db.commit()
            self.db.refresh(reading)
            return reading
        except Exception:
            self.db.rollback()
            raise

    def get_reading(self, reading_id: int) -> Optional[FaceReading]:
        return self.db.query(FaceReading).filter(FaceReading.id == reading_id).first()

    def get_user_readings(self, user_id: int, skip: int = 0, limit: int = 100) -> List[FaceReading]:
        return self.db.query(FaceReading).filter(
            FaceReading.user_id == user_id
        ).order_by(FaceReading.created_at.desc()).offset(skip).limit(limit).all()


class BaziService:
    def __init__(self, db: Session):
        self.db = db

    def create_bazi_reading(self, data: dict) -> BaziReading:
        try:
            reading = BaziReading(
                user_id=data.get("user_id"),
                birth_year=data.get("birth_year"),
                birth_month=data.get("birth_month"),
                birth_day=data.get("birth_day"),
                birth_hour=data.get("birth_hour"),
                year_pillar=data.get("year_pillar"),
                month_pillar=data.get("month_pillar"),
                day_pillar=data.get("day_pillar"),
                hour_pillar=data.get("hour_pillar"),
                five_elements=data.get("five_elements"),
                analysis=data.get("analysis"),
                created_at=datetime.now()
            )
            self.db.add(reading)
            self.db.commit()
            self.db.refresh(reading)
            return reading
        except Exception:
            self.db.rollback()
            raise

    def get_reading(self, reading_id: int) -> Optional[BaziReading]:
        return self.db.query(BaziReading).filter(BaziReading.id == reading_id).first()

    def get_user_readings(self, user_id: int, skip: int = 0, limit: int = 100) -> List[BaziReading]:
        return self.db.query(BaziReading).filter(
            BaziReading.user_id == user_id
        ).order_by(BaziReading.created_at.desc()).offset(skip).limit(limit).all()


class ZhouyiService:
    def __init__(self, db: Session):
        self.db = db

    def create_zhouyi_reading(self, data: dict) -> ZhouyiReading:
        try:
            reading = ZhouyiReading(
                user_id=data.get("user_id"),
                question=data.get("question"),
                hexagram=data.get("hexagram"),
                changing_lines=data.get("changing_lines"),
                result_hexagram=data.get("result_hexagram"),
                interpretation=data.get("interpretation"),
                advice=data.get("advice"),
                created_at=datetime.now()
            )
            self.db.add(reading)
            self.db.commit()
            self.db.refresh(reading)
            return reading
        except Exception:
            self.db.rollback()
            raise

    def get_reading(self, reading_id: int) -> Optional[ZhouyiReading]:
        return self.db.query(ZhouyiReading).filter(ZhouyiReading.id == reading_id).first()

    def get_user_readings(self, user_id: int, skip: int = 0, limit: int = 100) -> List[ZhouyiReading]:
        return self.db.query(ZhouyiReading).filter(
            ZhouyiReading.user_id == user_id
        ).order_by(ZhouyiReading.created_at.desc()).offset(skip).limit(limit).all()


class ConstellationService:
    def __init__(self, db: Session):
        self.db = db

    def create_constellation(self, data: dict) -> Constellation:
        try:
            constellation = Constellation(
                user_id=data.get("user_id"),
                zodiac_sign=data.get("zodiac_sign"),
                date=data.get("date"),
                overall_score=data.get("overall_score"),
                love_score=data.get("love_score"),
                career_score=data.get("career_score"),
                health_score=data.get("health_score"),
                wealth_score=data.get("wealth_score"),
                advice=data.get("advice"),
                created_at=datetime.now()
            )
            self.db.add(constellation)
            self.db.commit()
            self.db.refresh(constellation)
            return constellation
        except Exception:
            self.db.rollback()
            raise

    def get_constellation(self, constellation_id: int) -> Optional[Constellation]:
        return self.db.query(Constellation).filter(Constellation.id == constellation_id).first()

    def get_user_constellations(self, user_id: int, skip: int = 0, limit: int = 100) -> List[Constellation]:
        return self.db.query(Constellation).filter(
            Constellation.user_id == user_id
        ).order_by(Constellation.created_at.desc()).offset(skip).limit(limit).all()


class DailyFortuneService:
    def __init__(self, db: Session):
        self.db = db

    def create_daily_fortune(self, data: dict) -> DailyFortune:
        try:
            fortune = DailyFortune(
                user_id=data.get("user_id"),
                fortune_date=data.get("fortune_date"),
                overall_score=data.get("overall_score"),
                love_fortune=data.get("love_fortune"),
                career_fortune=data.get("career_fortune"),
                health_fortune=data.get("health_fortune"),
                wealth_fortune=data.get("wealth_fortune"),
                lucky_number=data.get("lucky_number"),
                lucky_color=data.get("lucky_color"),
                advice=data.get("advice"),
                created_at=datetime.now()
            )
            self.db.add(fortune)
            self.db.commit()
            self.db.refresh(fortune)
            return fortune
        except Exception:
            self.db.rollback()
            raise

    def get_fortune(self, fortune_id: int) -> Optional[DailyFortune]:
        return self.db.query(DailyFortune).filter(DailyFortune.id == fortune_id).first()

    def get_user_fortunes(self, user_id: int, skip: int = 0, limit: int = 100) -> List[DailyFortune]:
        return self.db.query(DailyFortune).filter(
            DailyFortune.user_id == user_id
        ).order_by(DailyFortune.created_at.desc()).offset(skip).limit(limit).all()
