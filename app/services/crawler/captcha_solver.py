"""验证码识别模块

此模块提供验证码的识别和处理功能。
"""
import base64
import logging
import time
from typing import Optional, Dict, Any
from pathlib import Path

from app.core.logger import get_logger

logger = get_logger("captcha_solver")


class CaptchaSolver:
    """验证码识别器"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self.retry_count = 3
        self.timeout = 30
    
    def solve_image_captcha(self, image_data: bytes, captcha_type: str = "default") -> Optional[str]:
        """识别图片验证码"""
        try:
            # 这里可以集成第三方验证码识别服务
            # 如：打码平台、OCR 服务等
            
            # 示例：使用本地 OCR（需要安装 pytesseract）
            try:
                import pytesseract
                from PIL import Image
                import io
                
                image = Image.open(io.BytesIO(image_data))
                text = pytesseract.image_to_string(image, lang='eng+chi_sim')
                result = text.strip()
                
                logger.info(f"验证码识别结果：{result}")
                return result if result else None
                
            except ImportError:
                logger.warning("未安装 pytesseract，无法识别验证码")
                return None
            
        except Exception as e:
            logger.error(f"验证码识别失败：{e}")
            return None
    
    def solve_slider_captcha(self, slide_data: Dict[str, Any]) -> Optional[int]:
        """识别滑块验证码"""
        try:
            # 滑块验证码需要计算缺口位置
            # 这里提供基本的处理框架
            
            if 'background_image' in slide_data and 'slide_image' in slide_data:
                # 使用图像处理计算缺口
                gap_position = self._calculate_gap(
                    slide_data['background_image'],
                    slide_data['slide_image']
                )
                return gap_position
            
            return None
            
        except Exception as e:
            logger.error(f"滑块验证码识别失败：{e}")
            return None
    
    def _calculate_gap(self, background: bytes, slide: bytes) -> Optional[int]:
        """计算滑块缺口位置"""
        try:
            import cv2
            import numpy as np
            from PIL import Image
            import io
            
            # 转换为 OpenCV 格式
            bg_img = cv2.cvtColor(np.array(Image.open(io.BytesIO(background))), cv2.COLOR_RGB2BGR)
            slide_img = cv2.cvtColor(np.array(Image.open(io.BytesIO(slide))), cv2.COLOR_RGB2BGR)
            
            # 边缘检测
            bg_edge = cv2.Canny(bg_img, 100, 200)
            slide_edge = cv2.Canny(slide_img, 100, 200)
            
            # 模板匹配
            result = cv2.matchTemplate(bg_edge, slide_edge, cv2.TM_CCOEFF_NORMED)
            _, _, _, max_loc = cv2.minMaxLoc(result)
            
            gap_x = max_loc[0]
            logger.info(f"滑块缺口位置：{gap_x}")
            
            return gap_x
            
        except ImportError:
            logger.warning("未安装 OpenCV，无法识别滑块验证码")
            return None
        except Exception as e:
            logger.error(f"计算滑块缺口失败：{e}")
            return None
    
    def solve_click_captcha(self, image_data: bytes, click_type: str = "text") -> Optional[list]:
        """识别点击验证码"""
        try:
            # 点击验证码需要识别特定文字或物体的位置
            # 这里提供基本的处理框架
            
            if click_type == "text":
                # 识别文字位置
                return self._find_text_positions(image_data)
            elif click_type == "object":
                # 识别物体位置
                return self._find_object_positions(image_data)
            
            return None
            
        except Exception as e:
            logger.error(f"点击验证码识别失败：{e}")
            return None
    
    def _find_text_positions(self, image_data: bytes) -> Optional[list]:
        """查找文字位置"""
        try:
            import cv2
            import numpy as np
            from PIL import Image
            import io
            
            image = cv2.cvtColor(np.array(Image.open(io.BytesIO(image_data))), cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # 二值化
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # 查找轮廓
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            positions = []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w > 20 and h > 20:  # 过滤太小的区域
                    positions.append([x, y])
            
            logger.info(f"找到 {len(positions)} 个文字区域")
            return positions
            
        except Exception as e:
            logger.error(f"查找文字位置失败：{e}")
            return None
    
    def _find_object_positions(self, image_data: bytes) -> Optional[list]:
        """查找物体位置"""
        # 使用目标检测模型（如 YOLO）
        # 这里仅提供框架
        logger.warning("物体识别功能需要集成目标检测模型")
        return None


class CaptchaHandler:
    """验证码处理器"""
    
    def __init__(self, solver: Optional[CaptchaSolver] = None):
        self.solver = solver or CaptchaSolver()
        self.captcha_count = 0
        self.success_count = 0
    
    async def handle_captcha(self, captcha_info: Dict[str, Any]) -> Optional[str]:
        """处理验证码"""
        try:
            captcha_type = captcha_info.get("type", "image")
            
            if captcha_type == "image":
                return await self._handle_image_captcha(captcha_info)
            elif captcha_type == "slider":
                return await self._handle_slider_captcha(captcha_info)
            elif captcha_type == "click":
                return await self._handle_click_captcha(captcha_info)
            else:
                logger.warning(f"未知的验证码类型：{captcha_type}")
                return None
                
        except Exception as e:
            logger.error(f"处理验证码失败：{e}")
            return None
    
    async def _handle_image_captcha(self, captcha_info: Dict[str, Any]) -> Optional[str]:
        """处理图片验证码"""
        try:
            image_data = captcha_info.get("image_data")
            if not image_data:
                logger.warning("验证码图片数据为空")
                return None
            
            # 如果是 base64 编码，解码
            if isinstance(image_data, str) and image_data.startswith("data:"):
                image_data = base64.b64decode(image_data.split(",")[1])
            
            # 识别验证码
            result = self.solver.solve_image_captcha(image_data)
            
            if result:
                self.success_count += 1
                logger.info(f"图片验证码识别成功：{result}")
            
            self.captcha_count += 1
            return result
            
        except Exception as e:
            logger.error(f"处理图片验证码失败：{e}")
            return None
    
    async def _handle_slider_captcha(self, captcha_info: Dict[str, Any]) -> Optional[int]:
        """处理滑块验证码"""
        try:
            slide_data = {
                "background_image": captcha_info.get("background_image"),
                "slide_image": captcha_info.get("slide_image")
            }
            
            # 识别滑块位置
            gap_position = self.solver.solve_slider_captcha(slide_data)
            
            if gap_position is not None:
                self.success_count += 1
                logger.info(f"滑块验证码识别成功，缺口位置：{gap_position}")
            
            self.captcha_count += 1
            return gap_position
            
        except Exception as e:
            logger.error(f"处理滑块验证码失败：{e}")
            return None
    
    async def _handle_click_captcha(self, captcha_info: Dict[str, Any]) -> Optional[list]:
        """处理点击验证码"""
        try:
            image_data = captcha_info.get("image_data")
            click_type = captcha_info.get("click_type", "text")
            
            if not image_data:
                logger.warning("验证码图片数据为空")
                return None
            
            # 识别点击位置
            positions = self.solver.solve_click_captcha(image_data, click_type)
            
            if positions:
                self.success_count += 1
                logger.info(f"点击验证码识别成功，找到 {len(positions)} 个位置")
            
            self.captcha_count += 1
            return positions
            
        except Exception as e:
            logger.error(f"处理点击验证码失败：{e}")
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        success_rate = self.success_count / max(1, self.captcha_count)
        return {
            "total_captchas": self.captcha_count,
            "success_count": self.success_count,
            "success_rate": round(success_rate, 2)
        }


# 全局验证码处理器
_captcha_handler: Optional[CaptchaHandler] = None


def get_captcha_handler() -> CaptchaHandler:
    """获取验证码处理器实例"""
    global _captcha_handler
    
    if _captcha_handler is None:
        _captcha_handler = CaptchaHandler()
    
    return _captcha_handler
