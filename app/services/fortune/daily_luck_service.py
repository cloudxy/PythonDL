"""每日运势预测服务模块

此模块提供每日运势预测功能。
"""
from datetime import datetime, date, timedelta
from typing import Dict, List
import random
import logging

logger = logging.getLogger(__name__)


class DailyLuckService:
    """每日运势服务"""
    
    # 十二生肖
    ZODIACS = ['鼠', '牛', '虎', '兔', '龙', '蛇', '马', '羊', '猴', '鸡', '狗', '猪']
    
    # 十二星座
    CONSTELLATIONS = [
        '白羊座', '金牛座', '双子座', '巨蟹座',
        '狮子座', '处女座', '天秤座', '天蝎座',
        '射手座', '摩羯座', '水瓶座', '双鱼座'
    ]
    
    # 吉凶等级
    LUCK_LEVELS = {
        '大吉': '非常吉利，诸事顺利',
        '吉': '吉利，多数事情顺利',
        '中吉': '中等吉利，需要注意',
        '小吉': '小有吉利，谨慎行事',
        '平': '平淡，无特别吉凶',
        '小凶': '小有不利，需要小心',
        '凶': '不利，尽量避免重要决策',
        '大凶': '非常不利，宜静不宜动'
    }
    
    # 方位
    DIRECTIONS = ['东方', '南方', '西方', '北方', '东南', '西南', '西北', '东北']
    
    # 宜忌
    YI_ITEMS = [
        '出行', '签约', '交易', '求财', '祭祀', '祈福', '嫁娶', '动土',
        '开市', '纳财', '安床', '修造', '破土', '安葬', '入宅', '移徙'
    ]
    
    JI_ITEMS = [
        '诉讼', '词讼', '行舟', '破财', '嫁娶', '出行', '动土', '开仓',
        '探病', '安葬', '修造', '入宅', '移徙', '祈福', '祭祀', '纳财'
    ]
    
    def __init__(self):
        pass
    
    def get_daily_luck(self, target_date: date = None, 
                      zodiac: str = None, 
                      constellation: str = None) -> Dict:
        """
        获取每日运势
        
        Args:
            target_date: 日期，默认为今天
            zodiac: 生肖
            constellation: 星座
            
        Returns:
            运势信息
        """
        if target_date is None:
            target_date = date.today()
        
        # 计算日期干支
        gan_zhi = self._calculate_date_gan_zhi(target_date)
        
        # 生肖运势
        if zodiac:
            zodiac_luck = self._get_zodiac_luck(zodiac, target_date)
        else:
            zodiac_luck = {z: self._get_zodiac_luck(z, target_date) for z in self.ZODIACS}
        
        # 星座运势
        if constellation:
            constellation_luck = self._get_constellation_luck(constellation, target_date)
        else:
            constellation_luck = {c: self._get_constellation_luck(c, target_date) 
                                 for c in self.CONSTELLATIONS}
        
        # 吉凶
        luck_level = self._calculate_luck_level(gan_zhi, target_date)
        
        # 宜忌
        yi_ji = self._calculate_yi_ji(target_date)
        
        # 财神方位
        wealth_direction = self._calculate_wealth_direction(gan_zhi)
        
        # 幸运数字和颜色
        lucky_elements = self._get_lucky_elements(target_date)
        
        return {
            'date': target_date.isoformat(),
            'gan_zhi': gan_zhi,
            'zodiac_luck': zodiac_luck,
            'constellation_luck': constellation_luck,
            'luck_level': luck_level,
            'yi_ji': yi_ji,
            'wealth_direction': wealth_direction,
            'lucky_elements': lucky_elements
        }
    
    def _calculate_date_gan_zhi(self, dt: date) -> Dict:
        """计算日期干支"""
        # 简化计算
        tian_gan = ['甲', '乙', '丙', '丁', '戊', '己', '庚', '辛', '壬', '癸']
        di_zhi = ['子', '丑', '寅', '卯', '辰', '巳', '午', '未', '申', '酉', '戌', '亥']
        
        # 基准日 2000 年 1 月 1 日
        base_date = date(2000, 1, 1)
        days_diff = (dt - base_date).days
        
        gan_index = days_diff % 10
        zhi_index = days_diff % 12
        
        return {
            'year_gan': tian_gan[dt.year % 10],
            'year_zhi': di_zhi[dt.year % 12],
            'day_gan': tian_gan[gan_index],
            'day_zhi': di_zhi[zhi_index]
        }
    
    def _get_zodiac_luck(self, zodiac: str, dt: date) -> Dict:
        """获取生肖运势"""
        # 使用日期和生肖生成运势
        seed = hash(f"{zodiac}{dt.isoformat()}") % 100
        random.seed(seed)
        
        luck_level = random.choice(list(self.LUCK_LEVELS.keys()))
        score = random.randint(60, 95)
        
        return {
            'zodiac': zodiac,
            'luck_level': luck_level,
            'score': score,
            'advice': f"{zodiac}今日运势：{self.LUCK_LEVELS[luck_level]}"
        }
    
    def _get_constellation_luck(self, constellation: str, dt: date) -> Dict:
        """获取星座运势"""
        seed = hash(f"{constellation}{dt.isoformat()}") % 100
        random.seed(seed)
        
        luck_level = random.choice(list(self.LUCK_LEVELS.keys()))
        score = random.randint(60, 95)
        
        love_score = random.randint(50, 100)
        career_score = random.randint(50, 100)
        wealth_score = random.randint(50, 100)
        health_score = random.randint(50, 100)
        
        return {
            'constellation': constellation,
            'luck_level': luck_level,
            'score': score,
            'love': {'score': love_score, 'advice': self._get_love_advice(love_score)},
            'career': {'score': career_score, 'advice': self._get_career_advice(career_score)},
            'wealth': {'score': wealth_score, 'advice': self._get_wealth_advice(wealth_score)},
            'health': {'score': health_score, 'advice': self._get_health_advice(health_score)}
        }
    
    def _calculate_luck_level(self, gan_zhi: Dict, dt: date) -> str:
        """计算吉凶"""
        seed = hash(f"{gan_zhi}{dt.isoformat()}") % 100
        random.seed(seed)
        
        return random.choice(list(self.LUCK_LEVELS.keys()))
    
    def _calculate_yi_ji(self, dt: date) -> Dict:
        """计算宜忌"""
        seed = hash(f"yiji{dt.isoformat()}") % 100
        random.seed(seed)
        
        yi_count = random.randint(3, 6)
        ji_count = random.randint(3, 6)
        
        yi = random.sample(self.YI_ITEMS, yi_count)
        ji = random.sample(self.JI_ITEMS, ji_count)
        
        return {
            'yi': yi,
            'ji': ji
        }
    
    def _calculate_wealth_direction(self, gan_zhi: Dict) -> str:
        """计算财神方位"""
        day_gan = gan_zhi.get('day_gan', '甲')
        
        direction_map = {
            '甲': '东南', '乙': '东南',
            '丙': '正东', '丁': '正东',
            '戊': '正北', '己': '正北',
            '庚': '正南', '辛': '正南',
            '壬': '东南', '癸': '东南'
        }
        
        return direction_map.get(day_gan, '东南')
    
    def _get_lucky_elements(self, dt: date) -> Dict:
        """获取幸运元素"""
        seed = hash(f"lucky{dt.isoformat()}") % 100
        random.seed(seed)
        
        colors = ['红色', '黄色', '绿色', '蓝色', '白色', '黑色', '紫色', '粉色']
        numbers = list(range(1, 10))
        
        return {
            'colors': random.sample(colors, 2),
            'numbers': random.sample(numbers, 2),
            'flowers': random.choice(['玫瑰', '百合', '康乃馨', '向日葵'])
        }
    
    def _get_love_advice(self, score: int) -> str:
        """爱情建议"""
        if score >= 80:
            return "爱情运势很好，适合表白或增进感情"
        elif score >= 60:
            return "爱情运势平稳，多沟通理解"
        else:
            return "爱情运势一般，避免争吵"
    
    def _get_career_advice(self, score: int) -> str:
        """事业建议"""
        if score >= 80:
            return "事业运势旺盛，适合推进重要项目"
        elif score >= 60:
            return "事业运势平稳，按部就班即可"
        else:
            return "事业运势欠佳，谨慎决策"
    
    def _get_wealth_advice(self, score: int) -> str:
        """财富建议"""
        if score >= 80:
            return "财运亨通，可能有意外之财"
        elif score >= 60:
            return "财运平稳，正财为主"
        else:
            return "财运不佳，避免投资"
    
    def _get_health_advice(self, score: int) -> str:
        """健康建议"""
        if score >= 80:
            return "健康状况良好，精力充沛"
        elif score >= 60:
            return "健康状况平稳，注意休息"
        else:
            return "健康状况欠佳，注意保养"


daily_luck_service = DailyLuckService()
