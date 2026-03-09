"""create fortune_telling table

Revision ID: create_fortune_telling
Revises: create_weather_forecasts
Create Date: 2024-01-01 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from datetime import datetime


revision = 'create_fortune_telling'
down_revision = 'create_weather_forecasts'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'fortune_telling',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False, comment='主键ID'),
        sa.Column('user_id', sa.String(length=50), nullable=False, comment='用户ID'),
        sa.Column('session_id', sa.String(length=100), nullable=False, comment='会话ID'),
        sa.Column('fortune_type', sa.String(length=50), nullable=False, comment='算命类型'),
        sa.Column('birth_date', sa.DateTime(), nullable=True, comment='出生日期'),
        sa.Column('birth_time', sa.String(length=20), nullable=True, comment='出生时辰'),
        sa.Column('gender', sa.String(length=10), nullable=True, comment='性别'),
        sa.Column('name', sa.String(length=50), nullable=True, comment='姓名'),
        sa.Column('input_data', sa.JSON(), nullable=True, comment='输入数据'),
        sa.Column('result_data', sa.JSON(), nullable=True, comment='结果数据'),
        sa.Column('overall_score', sa.Float(), nullable=True, comment='综合评分'),
        sa.Column('luck_score', sa.Float(), nullable=True, comment='运势评分'),
        sa.Column('career_score', sa.Float(), nullable=True, comment='事业评分'),
        sa.Column('love_score', sa.Float(), nullable=True, comment='爱情评分'),
        sa.Column('wealth_score', sa.Float(), nullable=True, comment='财富评分'),
        sa.Column('health_score', sa.Float(), nullable=True, comment='健康评分'),
        sa.Column('analysis_text', sa.Text(), nullable=True, comment='分析文本'),
        sa.Column('advice_text', sa.Text(), nullable=True, comment='建议文本'),
        sa.Column('created_at', sa.DateTime(), default=datetime.now, nullable=True, comment='创建时间'),
        sa.Column('updated_at', sa.DateTime(), default=datetime.now, onupdate=datetime.now, nullable=True, comment='更新时间'),
        sa.PrimaryKeyConstraint('id'),
        comment='看相算命数据表'
    )
    op.create_index('idx_fortune_user_time', 'fortune_telling', ['user_id', 'created_at'])
    op.create_index('idx_fortune_type_time', 'fortune_telling', ['fortune_type', 'created_at'])
    op.create_index('ix_fortune_telling_user_id', 'fortune_telling', ['user_id'])
    op.create_index('ix_fortune_telling_session_id', 'fortune_telling', ['session_id'])
    op.create_index('ix_fortune_telling_fortune_type', 'fortune_telling', ['fortune_type'])


def downgrade():
    op.drop_index('ix_fortune_telling_fortune_type', table_name='fortune_telling')
    op.drop_index('ix_fortune_telling_session_id', table_name='fortune_telling')
    op.drop_index('ix_fortune_telling_user_id', table_name='fortune_telling')
    op.drop_index('idx_fortune_type_time', table_name='fortune_telling')
    op.drop_index('idx_fortune_user_time', table_name='fortune_telling')
    op.drop_table('fortune_telling')
