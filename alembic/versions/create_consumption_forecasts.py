"""create consumption_forecasts table

Revision ID: create_consumption_forecasts
Revises: create_consumption_categories
Create Date: 2024-01-01 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from datetime import datetime


revision = 'create_consumption_forecasts'
down_revision = 'create_consumption_categories'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'consumption_forecasts',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False, comment='主键ID'),
        sa.Column('region', sa.String(length=50), nullable=False, comment='地区'),
        sa.Column('forecast_date', sa.DateTime(), nullable=False, comment='预测日期'),
        sa.Column('valid_date', sa.DateTime(), nullable=False, comment='有效日期'),
        sa.Column('forecast_type', sa.String(length=20), nullable=True, comment='预测类型'),
        sa.Column('total_consumption', sa.Float(), nullable=True, comment='总消费额预测'),
        sa.Column('per_capita_consumption', sa.Float(), nullable=True, comment='人均消费额预测'),
        sa.Column('category_forecasts', sa.JSON(), nullable=True, comment='类别消费预测'),
        sa.Column('growth_rate', sa.Float(), nullable=True, comment='增长率预测'),
        sa.Column('confidence', sa.Float(), nullable=True, comment='预测置信度'),
        sa.Column('model_version', sa.String(length=50), nullable=True, comment='模型版本'),
        sa.Column('created_at', sa.DateTime(), default=datetime.now, nullable=True, comment='创建时间'),
        sa.Column('updated_at', sa.DateTime(), default=datetime.now, onupdate=datetime.now, nullable=True, comment='更新时间'),
        sa.PrimaryKeyConstraint('id'),
        comment='消费预测数据表'
    )
    op.create_index('idx_consumption_forecast_region_date', 'consumption_forecasts', ['region', 'forecast_date'])
    op.create_index('idx_consumption_forecast_valid_date', 'consumption_forecasts', ['valid_date'])
    op.create_index('ix_consumption_forecasts_region', 'consumption_forecasts', ['region'])


def downgrade():
    op.drop_index('ix_consumption_forecasts_region', table_name='consumption_forecasts')
    op.drop_index('idx_consumption_forecast_valid_date', table_name='consumption_forecasts')
    op.drop_index('idx_consumption_forecast_region_date', table_name='consumption_forecasts')
    op.drop_table('consumption_forecasts')
