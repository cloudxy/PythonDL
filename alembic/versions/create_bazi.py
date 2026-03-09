"""create bazi table

Revision ID: create_bazi
Revises: create_fortune_telling
Create Date: 2024-01-01 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from datetime import datetime


revision = 'create_bazi'
down_revision = 'create_fortune_telling'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'bazi',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False, comment='主键ID'),
        sa.Column('user_id', sa.String(length=50), nullable=False, comment='用户ID'),
        sa.Column('session_id', sa.String(length=100), nullable=False, comment='会话ID'),
        sa.Column('birth_date', sa.DateTime(), nullable=False, comment='出生日期'),
        sa.Column('birth_time', sa.String(length=20), nullable=False, comment='出生时辰'),
        sa.Column('gender', sa.String(length=10), nullable=False, comment='性别'),
        sa.Column('year_gan', sa.String(length=10), nullable=True, comment='年干'),
        sa.Column('year_zhi', sa.String(length=10), nullable=True, comment='年支'),
        sa.Column('month_gan', sa.String(length=10), nullable=True, comment='月干'),
        sa.Column('month_zhi', sa.String(length=10), nullable=True, comment='月支'),
        sa.Column('day_gan', sa.String(length=10), nullable=True, comment='日干'),
        sa.Column('day_zhi', sa.String(length=10), nullable=True, comment='日支'),
        sa.Column('hour_gan', sa.String(length=10), nullable=True, comment='时干'),
        sa.Column('hour_zhi', sa.String(length=10), nullable=True, comment='时支'),
        sa.Column('day_master', sa.String(length=10), nullable=True, comment='日主'),
        sa.Column('five_elements', sa.JSON(), nullable=True, comment='五行分布'),
        sa.Column('ten_gods', sa.JSON(), nullable=True, comment='十神分布'),
        sa.Column('strength', sa.String(length=20), nullable=True, comment='身强身弱'),
        sa.Column('favorable_elements', sa.JSON(), nullable=True, comment='喜用神'),
        sa.Column('unfavorable_elements', sa.JSON(), nullable=True, comment='忌神'),
        sa.Column('analysis_result', sa.JSON(), nullable=True, comment='分析结果'),
        sa.Column('career_analysis', sa.Text(), nullable=True, comment='事业分析'),
        sa.Column('wealth_analysis', sa.Text(), nullable=True, comment='财富分析'),
        sa.Column('love_analysis', sa.Text(), nullable=True, comment='爱情分析'),
        sa.Column('health_analysis', sa.Text(), nullable=True, comment='健康分析'),
        sa.Column('created_at', sa.DateTime(), default=datetime.now, nullable=True, comment='创建时间'),
        sa.Column('updated_at', sa.DateTime(), default=datetime.now, onupdate=datetime.now, nullable=True, comment='更新时间'),
        sa.PrimaryKeyConstraint('id'),
        comment='八字算命数据表'
    )
    op.create_index('idx_bazi_user_time', 'bazi', ['user_id', 'created_at'])
    op.create_index('idx_bazi_birth', 'bazi', ['birth_date', 'birth_time'])
    op.create_index('ix_bazi_user_id', 'bazi', ['user_id'])
    op.create_index('ix_bazi_session_id', 'bazi', ['session_id'])


def downgrade():
    op.drop_index('ix_bazi_session_id', table_name='bazi')
    op.drop_index('ix_bazi_user_id', table_name='bazi')
    op.drop_index('idx_bazi_birth', table_name='bazi')
    op.drop_index('idx_bazi_user_time', table_name='bazi')
    op.drop_table('bazi')
