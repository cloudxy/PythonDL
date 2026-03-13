"""邮件服务模块

此模块提供邮件发送功能，用于密码找回、系统通知等。
"""
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.header import Header
from typing import Optional, List
from pathlib import Path
import logging

from app.core.config import config

logger = logging.getLogger(__name__)


class EmailService:
    """邮件服务类"""
    
    def __init__(self):
        self.smtp_host = config.SMTP_HOST
        self.smtp_port = config.SMTP_PORT
        self.smtp_user = config.SMTP_USER
        self.smtp_password = config.SMTP_PASSWORD
        self.from_email = config.FROM_EMAIL
        self.from_name = config.FROM_NAME
        self.use_tls = config.SMTP_USE_TLS
    
    def send_email(
        self,
        to_email: str,
        subject: str,
        html_content: str,
        text_content: Optional[str] = None
    ) -> bool:
        """发送邮件
        
        Args:
            to_email: 收件人邮箱
            subject: 邮件主题
            html_content: HTML 内容
            text_content: 纯文本内容（可选）
            
        Returns:
            bool: 是否发送成功
        """
        try:
            # 创建邮件
            msg = MIMEMultipart('alternative')
            msg['Subject'] = Header(subject, 'utf-8')
            msg['From'] = f"{self.from_name} <{self.from_email}>"
            msg['To'] = to_email
            
            # 添加纯文本版本
            if text_content:
                text_part = MIMEText(text_content, 'plain', 'utf-8')
                msg.attach(text_part)
            
            # 添加 HTML 版本
            html_part = MIMEText(html_content, 'html', 'utf-8')
            msg.attach(html_part)
            
            # 发送邮件
            if self.use_tls:
                server = smtplib.SMTP_SSL(self.smtp_host, self.smtp_port)
            else:
                server = smtplib.SMTP(self.smtp_host, self.smtp_port)
            
            if self.smtp_user and self.smtp_password:
                server.login(self.smtp_user, self.smtp_password)
            
            server.sendmail(self.from_email, [to_email], msg.as_string())
            server.quit()
            
            logger.info(f"邮件发送成功：{to_email}")
            return True
            
        except Exception as e:
            logger.error(f"邮件发送失败：{to_email}, 错误：{str(e)}")
            return False
    
    def send_password_reset_email(self, to_email: str, reset_token: str, username: str) -> bool:
        """发送密码重置邮件
        
        Args:
            to_email: 收件人邮箱
            reset_token: 重置令牌
            username: 用户名
            
        Returns:
            bool: 是否发送成功
        """
        subject = f"{self.from_name} - 密码重置"
        
        reset_link = f"{config.FRONTEND_URL}/reset-password?token={reset_token}"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; text-align: center; }}
                .content {{ padding: 30px; background: #f9f9f9; }}
                .button {{ display: inline-block; padding: 12px 30px; background: #667eea; color: white; text-decoration: none; border-radius: 5px; margin-top: 20px; }}
                .footer {{ text-align: center; padding: 20px; color: #666; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>密码重置</h1>
                </div>
                <div class="content">
                    <p>尊敬的 {username}：</p>
                    <p>您请求重置密码。请点击下方按钮重置您的密码：</p>
                    <p style="text-align: center;">
                        <a href="{reset_link}" class="button">重置密码</a>
                    </p>
                    <p>或者复制以下链接到浏览器：</p>
                    <p style="word-break: break-all; color: #667eea;">{reset_link}</p>
                    <p><strong>注意：</strong>此链接将在 24 小时后失效。</p>
                    <p>如果您没有请求重置密码，请忽略此邮件。</p>
                </div>
                <div class="footer">
                    <p>&copy; 2026 {self.from_name}. All rights reserved.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        text_content = f"""
        尊敬的 {username}：
        
        您请求重置密码。请访问以下链接重置您的密码：
        {reset_link}
        
        注意：此链接将在 24 小时后失效。
        
        如果您没有请求重置密码，请忽略此邮件。
        
        {self.from_name} 团队
        """
        
        return self.send_email(to_email, subject, html_content, text_content)
    
    def send_welcome_email(self, to_email: str, username: str) -> bool:
        """发送欢迎邮件
        
        Args:
            to_email: 收件人邮箱
            username: 用户名
            
        Returns:
            bool: 是否发送成功
        """
        subject = f"欢迎加入 {self.from_name}！"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; text-align: center; }}
                .content {{ padding: 30px; background: #f9f9f9; }}
                .footer {{ text-align: center; padding: 20px; color: #666; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>欢迎加入！</h1>
                </div>
                <div class="content">
                    <p>尊敬的 {username}：</p>
                    <p>感谢您注册 {self.from_name}！我们很高兴您加入我们。</p>
                    <p>您现在可以访问我们的平台，体验各种功能和服务。</p>
                    <p>如果您有任何问题，请随时联系我们的客服团队。</p>
                </div>
                <div class="footer">
                    <p>&copy; 2026 {self.from_name}. All rights reserved.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        return self.send_email(to_email, subject, html_content)
    
    def send_verification_email(self, to_email: str, username: str, verification_code: str) -> bool:
        """发送验证邮件
        
        Args:
            to_email: 收件人邮箱
            username: 用户名
            verification_code: 验证码
            
        Returns:
            bool: 是否发送成功
        """
        subject = f"{self.from_name} - 邮箱验证"
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <style>
                body {{ font-family: Arial, sans-serif; line-height: 1.6; }}
                .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
                .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 30px; text-align: center; }}
                .content {{ padding: 30px; background: #f9f9f9; }}
                .code {{ font-size: 32px; font-weight: bold; color: #667eea; text-align: center; padding: 20px; letter-spacing: 10px; }}
                .footer {{ text-align: center; padding: 20px; color: #666; font-size: 12px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>邮箱验证</h1>
                </div>
                <div class="content">
                    <p>尊敬的 {username}：</p>
                    <p>您的验证码是：</p>
                    <div class="code">{verification_code}</div>
                    <p>验证码有效期为 10 分钟。</p>
                    <p>如果您没有请求验证，请忽略此邮件。</p>
                </div>
                <div class="footer">
                    <p>&copy; 2026 {self.from_name}. All rights reserved.</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        text_content = f"""
        尊敬的 {username}：
        
        您的验证码是：{verification_code}
        
        验证码有效期为 10 分钟。
        
        如果您没有请求验证，请忽略此邮件。
        
        {self.from_name} 团队
        """
        
        return self.send_email(to_email, subject, html_content, text_content)


# 全局邮件服务实例
email_service = EmailService()
