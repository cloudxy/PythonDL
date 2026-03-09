"""create economic_indicators table

Revision ID: 20260211_1010_create_economic_indicators_table
Revises: 20260211_1009_create_consumption_data_table
Create Date: 2026-02-11 10:10:00.000000

"""
from alembic import op
import sqlalchemy as sa

revision = '20260211_1010_create_economic_indicators_table'
down_revision = '20260211_1009_create_consumption_data_table'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'economic_indicators',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False, comment='ID'),
        sa.Column('indicator_id', sa.String(length=50), nullable=False, comment='指标ID'),
        sa.Column('period', sa.String(length=20), nullable=False, comment='统计周期'),
        sa.Column('year', sa.Integer(), nullable=False, comment='年份'),
        sa.Column('quarter', sa.Integer(), nullable=True, comment='季度'),
        sa.Column('month', sa.Integer(), nullable=True, comment='月份'),
        sa.Column('indicator_name', sa.String(length=100), nullable=False, comment='指标名称'),
        sa.Column('indicator_value', sa.Float(), nullable=False, comment='指标值'),
        sa.Column('unit', sa.String(length=20), nullable=True, comment='单位'),
        sa.Column('region', sa.String(length=100), nullable=False, comment='地区'),
        sa.Column('growth_rate', sa.Float(), nullable=True, comment='增长率'),
        sa.Column('previous_value', sa.Float(), nullable=True, comment='上期值'),
        sa.Column('forecast_value', sa.Float(), nullable=True, comment='预测值'),
        sa.Column('source', sa.String(length=200), nullable=True, comment='数据来源'),
        sa.Column('description', sa.Text(), nullable=True, comment='指标描述'),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP'), comment='创建时间'),
        sa.Column('updated_at', sa.DateTime(), nullable=True, comment='更新时间'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_economic_indicators_indicator_id', 'economic_indicators', ['indicator_id'], unique=True)


def downgrade():
    op.drop_index('ix_economic_indicators_indicator_id', table_name='economic_indicators')
    op.drop_table('economic_indicators')