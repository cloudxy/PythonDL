"""create stocks table

Revision ID: 20260211_1005_create_stocks_table
Revises: 20260211_1004_create_stock_basics_table
Create Date: 2026-02-11 10:05:00.000000

"""
from alembic import op
import sqlalchemy as sa

revision = '20260211_1005_create_stocks_table'
down_revision = '20260211_1004_create_stock_basics_table'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'stocks',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False, comment='ID'),
        sa.Column('stock_code', sa.String(length=20), nullable=False, comment='股票代码'),
        sa.Column('stock_name', sa.String(length=100), nullable=False, comment='股票名称'),
        sa.Column('trade_date', sa.DateTime(), nullable=False, comment='交易日期'),
        sa.Column('open_price', sa.Float(), nullable=True, comment='开盘价'),
        sa.Column('close_price', sa.Float(), nullable=False, comment='收盘价'),
        sa.Column('high_price', sa.Float(), nullable=True, comment='最高价'),
        sa.Column('low_price', sa.Float(), nullable=True, comment='最低价'),
        sa.Column('volume', sa.BigInteger(), nullable=True, comment='成交量(股)'),
        sa.Column('amount', sa.Float(), nullable=True, comment='成交额(元)'),
        sa.Column('change_percent', sa.Float(), nullable=True, comment='涨跌幅(%)'),
        sa.Column('change_amount', sa.Float(), nullable=True, comment='涨跌额(元)'),
        sa.Column('turnover_rate', sa.Float(), nullable=True, comment='换手率(%)'),
        sa.Column('pe_ratio', sa.Float(), nullable=True, comment='市盈率'),
        sa.Column('pb_ratio', sa.Float(), nullable=True, comment='市净率'),
        sa.Column('total_market_cap', sa.Float(), nullable=True, comment='总市值(元)'),
        sa.Column('circulating_market_cap', sa.Float(), nullable=True, comment='流通市值(元)'),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP'), comment='创建时间'),
        sa.Column('updated_at', sa.DateTime(), nullable=True, comment='更新时间'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_stocks_stock_code', 'stocks', ['stock_code'])
    op.create_index('ix_stocks_trade_date', 'stocks', ['trade_date'])


def downgrade():
    op.drop_index('ix_stocks_trade_date', table_name='stocks')
    op.drop_index('ix_stocks_stock_code', table_name='stocks')
    op.drop_table('stocks')