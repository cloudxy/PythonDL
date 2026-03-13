"""八字排盘服务模块

此模块提供详细的八字排盘算法，包括天干地支、十神、大运等。
"""
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class BaziService:
    """八字排盘服务"""
    
    # 天干
    TIAN_GAN = ['甲', '乙', '丙', '丁', '戊', '己', '庚', '辛', '壬', '癸']
    
    # 地支
    DI_ZHI = ['子', '丑', '寅', '卯', '辰', '巳', '午', '未', '申', '酉', '戌', '亥']
    
    # 五行
    WU_XING = {
        '甲': '木', '乙': '木',
        '丙': '火', '丁': '火',
        '戊': '土', '己': '土',
        '庚': '金', '辛': '金',
        '壬': '水', '癸': '水',
        '子': '水', '亥': '水',
        '寅': '木', '卯': '木',
        '巳': '火', '午': '火',
        '申': '金', '酉': '金',
        '辰': '土', '戌': '土', '丑': '土', '未': '土'
    }
    
    # 十神关系
    SHI_SHEN_RELATIONS = {
        '比肩': '同五行同阴阳',
        '劫财': '同五行异阴阳',
        '食神': '我生者同阴阳',
        '伤官': '我生者异阴阳',
        '偏财': '我克者同阴阳',
        '正财': '我克者异阴阳',
        '七杀': '克我者同阴阳',
        '正官': '克我者异阴阳',
        '偏印': '生我者同阴阳',
        '正印': '生我者异阴阳'
    }
    
    def __init__(self):
        pass
    
    def calculate_bazi(self, birth_datetime: datetime, gender: str = 'male') -> Dict:
        """
        计算八字
        
        Args:
            birth_datetime: 出生时间
            gender: 性别 'male' 或 'female'
            
        Returns:
            八字信息字典
        """
        # 年柱
        year_pillar = self._get_year_pillar(birth_datetime)
        
        # 月柱
        month_pillar = self._get_month_pillar(birth_datetime, year_pillar['tian_gan'])
        
        # 日柱
        day_pillar = self._get_day_pillar(birth_datetime)
        
        # 时柱
        hour_pillar = self._get_hour_pillar(birth_datetime, day_pillar['tian_gan'])
        
        # 十神
        ten_gods = self._calculate_ten_gods(day_pillar['tian_gan'], [
            year_pillar, month_pillar, hour_pillar
        ])
        
        # 大运
        major_cycle = self._calculate_major_cycle(birth_datetime, gender, year_pillar)
        
        # 五行统计
        wu_xing_count = self._count_wu_xing([year_pillar, month_pillar, day_pillar, hour_pillar])
        
        # 五行旺衰
        wu_xing_strength = self._analyze_wu_xing_strength(wu_xing_count, birth_datetime)
        
        return {
            'year': year_pillar,
            'month': month_pillar,
            'day': day_pillar,
            'hour': hour_pillar,
            'ten_gods': ten_gods,
            'major_cycle': major_cycle,
            'wu_xing_count': wu_xing_count,
            'wu_xing_strength': wu_xing_strength,
            'day_master': day_pillar['tian_gan'],
            'day_master_wu_xing': self.WU_XING[day_pillar['tian_gan']]
        }
    
    def _get_year_pillar(self, dt: datetime) -> Dict:
        """获取年柱"""
        year = dt.year
        
        # 年干计算（年份 -3 后除以 10 的余数）
        gan_index = (year - 3) % 10
        
        # 年支计算（年份 -3 后除以 12 的余数）
        zhi_index = (year - 3) % 12
        
        tian_gan = self.TIAN_GAN[gan_index]
        di_zhi = self.DI_ZHI[zhi_index]
        
        return {
            'tian_gan': tian_gan,
            'di_zhi': di_zhi,
            'pillar': f'{tian_gan}{di_zhi}',
            'wu_xing': self.WU_XING[tian_gan] + self.WU_XING[di_zhi]
        }
    
    def _get_month_pillar(self, dt: datetime, year_gan: str) -> Dict:
        """获取月柱"""
        # 月支（农历月份）
        month = dt.month
        
        # 地支索引（寅月为正月）
        zhi_index = (month + 1) % 12
        if zhi_index == 0:
            zhi_index = 12
        zhi_index = zhi_index - 1
        
        di_zhi = self.DI_ZHI[zhi_index]
        
        # 月干计算（五虎遁元法）
        gan_map = {
            '甲': 2, '己': 2,  # 甲己之年丙作首
            '乙': 4, '庚': 4,  # 乙庚之岁戊为头
            '丙': 6, '辛': 6,  # 丙辛必定寻庚起
            '丁': 8, '壬': 8,  # 丁壬壬寅顺行流
            '戊': 0, '癸': 0   # 戊癸甲寅之上好追求
        }
        
        start_gan_index = gan_map.get(year_gan, 2)
        gan_index = (start_gan_index + month - 1) % 10
        tian_gan = self.TIAN_GAN[gan_index]
        
        return {
            'tian_gan': tian_gan,
            'di_zhi': di_zhi,
            'pillar': f'{tian_gan}{di_zhi}',
            'wu_xing': self.WU_XING[tian_gan] + self.WU_XING[di_zhi]
        }
    
    def _get_day_pillar(self, dt: datetime) -> Dict:
        """获取日柱"""
        # 简化计算，实际应该使用万年历
        # 这里使用一个基准日进行推算
        base_date = datetime(2000, 1, 1)  # 2000 年 1 月 1 日为甲午日
        days_diff = (dt - base_date).days
        
        # 60 甲子循环
        gan_zhi_index = days_diff % 60
        
        gan_index = gan_zhi_index % 10
        zhi_index = gan_zhi_index % 12
        
        tian_gan = self.TIAN_GAN[gan_index]
        di_zhi = self.DI_ZHI[zhi_index]
        
        return {
            'tian_gan': tian_gan,
            'di_zhi': di_zhi,
            'pillar': f'{tian_gan}{di_zhi}',
            'wu_xing': self.WU_XING[tian_gan] + self.WU_XING[di_zhi]
        }
    
    def _get_hour_pillar(self, dt: datetime, day_gan: str) -> Dict:
        """获取时柱"""
        hour = dt.hour
        
        # 时支计算（23-1 点为子时，1-3 点为丑时，以此类推）
        if hour >= 23:
            zhi_index = 0
        else:
            zhi_index = ((hour + 1) // 2) % 12
        
        di_zhi = self.DI_ZHI[zhi_index]
        
        # 时干计算（五鼠遁元法）
        gan_map = {
            '甲': 0, '己': 0,  # 甲己还加甲
            '乙': 2, '庚': 2,  # 乙庚丙作初
            '丙': 4, '辛': 4,  # 丙辛从戊起
            '丁': 6, '壬': 6,  # 丁壬庚子居
            '戊': 8, '癸': 8   # 戊癸壬子头
        }
        
        start_gan_index = gan_map.get(day_gan, 0)
        gan_index = (start_gan_index + zhi_index) % 10
        tian_gan = self.TIAN_GAN[gan_index]
        
        return {
            'tian_gan': tian_gan,
            'di_zhi': di_zhi,
            'pillar': f'{tian_gan}{di_zhi}',
            'wu_xing': self.WU_XING[tian_gan] + self.WU_XING[di_zhi]
        }
    
    def _calculate_ten_gods(self, day_gan: str, pillars: List[Dict]) -> Dict:
        """计算十神"""
        day_wu_xing = self.WU_XING[day_gan]
        day_yin_yang = 'yang' if self.TIAN_GAN.index(day_gan) % 2 == 0 else 'yin'
        
        ten_gods = {}
        
        for pillar in pillars:
            pillar_name = f"{pillar['pillar']}_{pillar['tian_gan']}"
            
            # 计算十神关系
            pillar_wu_xing = self.WU_XING[pillar['tian_gan']]
            pillar_yin_yang = 'yang' if self.TIAN_GAN.index(pillar['tian_gan']) % 2 == 0 else 'yin'
            
            shi_shen = self._determine_shi_shen(day_wu_xing, day_yin_yang, 
                                                 pillar_wu_xing, pillar_yin_yang)
            ten_gods[pillar_name] = shi_shen
        
        return ten_gods
    
    def _determine_shi_shen(self, day_wx: str, day_yy: str, 
                           pillar_wx: str, pillar_yy: str) -> str:
        """确定十神"""
        # 五行相生：木生火、火生土、土生金、金生水、水生木
        # 五行相克：木克土、土克水、水克火、火克金、金克木
        
        wu_xing_cycle = {'木': '火', '火': '土', '土': '金', '金': '水', '水': '木'}
        wu_xing_counter = {'木': '土', '土': '水', '水': '火', '火': '金', '金': '木'}
        
        same_yy = day_yy == pillar_yy
        
        if day_wx == pillar_wx:
            return '比肩' if same_yy else '劫财'
        elif wu_xing_cycle.get(day_wx) == pillar_wx:
            return '食神' if same_yy else '伤官'
        elif wu_xing_counter.get(day_wx) == pillar_wx:
            return '偏财' if same_yy else '正财'
        elif wu_xing_cycle.get(pillar_wx) == day_wx:
            return '偏印' if same_yy else '正印'
        elif wu_xing_counter.get(pillar_wx) == day_wx:
            return '七杀' if same_yy else '正官'
        
        return '未知'
    
    def _calculate_major_cycle(self, dt: datetime, gender: str, year_pillar: Dict) -> List[Dict]:
        """计算大运"""
        cycles = []
        
        # 起运岁数计算
        year_gan = year_pillar['tian_gan']
        year_yy = 'yang' if self.TIAN_GAN.index(year_gan) % 2 == 0 else 'yin'
        
        # 阳男阴女顺行，阴男阳女逆行
        if (gender == 'male' and year_yy == 'yang') or (gender == 'female' and year_yy == 'yin'):
            direction = 'forward'
        else:
            direction = 'backward'
        
        # 计算起运年龄（简化计算，实际应根据节气）
        start_age = 3  # 简化为 3 岁起运
        
        # 计算 8 步大运
        for i in range(8):
            age = start_age + i * 10
            
            if direction == 'forward':
                gan_index = (self.TIAN_GAN.index(year_pillar['tian_gan']) + i + 1) % 10
                zhi_index = (self.DI_ZHI.index(year_pillar['di_zhi']) + i + 1) % 12
            else:
                gan_index = (self.TIAN_GAN.index(year_pillar['tian_gan']) - i - 1) % 10
                zhi_index = (self.DI_ZHI.index(year_pillar['di_zhi']) - i - 1) % 12
            
            tian_gan = self.TIAN_GAN[gan_index]
            di_zhi = self.DI_ZHI[zhi_index]
            
            cycles.append({
                'age': age,
                'pillar': f'{tian_gan}{di_zhi}',
                'tian_gan': tian_gan,
                'di_zhi': di_zhi,
                'wu_xing': self.WU_XING[tian_gan] + self.WU_XING[di_zhi]
            })
        
        return cycles
    
    def _count_wu_xing(self, pillars: List[Dict]) -> Dict[str, int]:
        """统计五行数量"""
        count = {'金': 0, '木': 0, '水': 0, '火': 0, '土': 0}
        
        for pillar in pillars:
            for char in pillar['wu_xing']:
                if char in count:
                    count[char] += 1
        
        return count
    
    def _analyze_wu_xing_strength(self, wu_xing_count: Dict, dt: datetime) -> Dict:
        """分析五行旺衰"""
        # 季节旺相
        month = dt.month
        season_wu_xing = {
            (1, 2, 3): '木',   # 春季木旺
            (4, 5, 6): '火',   # 夏季火旺
            (7, 8, 9): '金',   # 秋季金旺
            (10, 11, 12): '水'  # 冬季水旺
        }
        
        season_wx = None
        for months, wx in season_wu_xing.items():
            if month in months:
                season_wx = wx
                break
        
        # 分析强弱
        strength = {}
        for wx, count in wu_xing_count.items():
            if count == 0:
                strength[wx] = '缺'
            elif count >= 3:
                strength[wx] = '旺'
            elif count == 2:
                strength[wx] = '平'
            else:
                strength[wx] = '弱'
        
        # 考虑季节因素
        if season_wx and strength.get(season_wx) in ['平', '弱']:
            strength[season_wx] = '相'
        
        return strength
    
    def get_fortune_analysis(self, bazi: Dict) -> Dict:
        """分析运势"""
        analysis = {
            'character': self._analyze_character(bazi),
            'career': self._analyze_career(bazi),
            'wealth': self._analyze_wealth(bazi),
            'marriage': self._analyze_marriage(bazi),
            'health': self._analyze_health(bazi),
            'lucky_elements': self._get_lucky_elements(bazi)
        }
        
        return analysis
    
    def _analyze_character(self, bazi: Dict) -> str:
        """分析性格"""
        day_master = bazi['day_master']
        return f"日主{day_master}，性格特征需要结合八字详细分析。"
    
    def _analyze_career(self, bazi: Dict) -> str:
        """分析事业"""
        return "事业运势需要结合大运流年综合分析。"
    
    def _analyze_wealth(self, bazi: Dict) -> str:
        """分析财运"""
        return "财运分析需要查看财星位置和旺衰。"
    
    def _analyze_marriage(self, bazi: Dict) -> str:
        """分析婚姻"""
        return "婚姻运势需要结合配偶宫分析。"
    
    def _analyze_health(self, bazi: Dict) -> str:
        """分析健康"""
        wu_xing = bazi['wu_xing_count']
        weak_elements = [wx for wx, count in wu_xing.items() if count == 0]
        
        if weak_elements:
            return f"五行缺{','.join(weak_elements)}，需要注意相关脏腑的保养。"
        return "五行相对平衡，注意日常保养即可。"
    
    def _get_lucky_elements(self, bazi: Dict) -> Dict:
        """获取幸运元素"""
        day_master = bazi['day_master']
        
        # 简化幸运元素计算
        lucky_colors = {
            '甲': ['绿色', '青色'], '乙': ['绿色', '青色'],
            '丙': ['红色', '紫色'], '丁': ['红色', '紫色'],
            '戊': ['黄色', '棕色'], '己': ['黄色', '棕色'],
            '庚': ['白色', '金色'], '辛': ['白色', '金色'],
            '壬': ['黑色', '蓝色'], '癸': ['黑色', '蓝色']
        }
        
        lucky_numbers = {
            '甲': [1, 8], '乙': [1, 8],
            '丙': [2, 7], '丁': [2, 7],
            '戊': [5, 0], '己': [5, 0],
            '庚': [4, 9], '辛': [4, 9],
            '壬': [3, 6], '癸': [3, 6]
        }
        
        return {
            'colors': lucky_colors.get(day_master, ['红色', '黄色']),
            'numbers': lucky_numbers.get(day_master, [1, 6]),
            'directions': ['东方', '南方']
        }


bazi_service = BaziService()
