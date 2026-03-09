"""create consumption_data table

Revision ID: 20260211_1009_create_consumption_data_table
Revises: 20260211_1008_create_fortune_tellings_table
Create Date: 2026-02-11 10:09:00.000000

"""
from alembic import op
import sqlalchemy as sa

revision = '20260211_1009_create_consumption_data_table'
down_revision = '20260211_1008_create_fortune_tellings_table'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'consumption_data',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False, comment='ID'),
        sa.Column('data_id', sa.String(length=50), nullable=False, comment='数据ID'),
        sa.Column('period', sa.String(length=20), nullable=False, comment='统计周期'),
        sa.Column('year', sa.Integer(), nullable=False, comment='年份'),
        sa.Column('quarter', sa.Integer(), nullable=True, comment='季度'),
        sa.Column('month', sa.Integer(), nullable=True, comment='月份'),
        sa.Column('region', sa.String(length=100), nullable=False, comment='地区'),
        sa.Column('consumption_type', sa.String(length=100), nullable=False, comment='消费类型'),
        sa.Column('total_consumption', sa.Float(), nullable=True, comment='总消费额'),
        sa.Column('per_capita_consumption', sa.Float(), nullable=True, comment='人均消费额'),
        sa.Column('urban_consumption', sa.Float(), nullable=True, comment='城镇消费额'),
        sa.Column('rural_consumption', sa.Float(), nullable=True, comment='农村消费额'),
        sa.Column('retail_sales', sa.Float(), nullable=True, comment='社会消费品零售总额'),
        sa.Column('online_retail_sales', sa.Float(), nullable=True, comment='网上零售额'),
        sa.Column('catering_industry_sales', sa.Float(), nullable=True, comment='餐饮业销售额'),
        sa.Column('consumer_confidence_index', sa.Float(), nullable=True, comment='消费者信心指数'),
        sa.Column('cpi', sa.Float(), nullable=True, comment='居民消费价格指数'),
        sa.Column('ppi', sa.Float(), nullable=True, comment='生产者价格指数'),
        sa.Column('description', sa.Text(), nullable=True, comment='数据描述'),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP'), comment='创建时间'),
        sa.Column('updated_at', sa.DateTime(), nullable=True, comment='更新时间'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_consumption_data_data_id', 'consumption_data', ['data_id'], unique=True)


def downgrade():
    op.drop_index('ix_consumption_data_data_id', table_name='consumption_data')
    op.drop_table('consumption_data')