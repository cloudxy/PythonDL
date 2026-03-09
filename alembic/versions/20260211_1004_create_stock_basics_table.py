"""create stock_basics table

Revision ID: 20260211_1004_create_stock_basics_table
Revises: 20260211_1003_create_role_permissions_table
Create Date: 2026-02-11 10:04:00.000000

"""
from alembic import op
import sqlalchemy as sa

revision = '20260211_1004_create_stock_basics_table'
down_revision = '20260211_1003_create_role_permissions_table'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'stock_basics',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False, comment='ID'),
        sa.Column('stock_code', sa.String(length=20), nullable=False, comment='股票代码'),
        sa.Column('stock_name', sa.String(length=100), nullable=False, comment='股票名称'),
        sa.Column('market', sa.String(length=20), nullable=False, comment='市场（沪市、深市、创业板等）'),
        sa.Column('industry', sa.String(length=100), nullable=True, comment='所属行业'),
        sa.Column('sector', sa.String(length=100), nullable=True, comment='所属板块'),
        sa.Column('list_date', sa.DateTime(), nullable=True, comment='上市日期'),
        sa.Column('delist_date', sa.DateTime(), nullable=True, comment='退市日期'),
        sa.Column('total_share', sa.Float(), nullable=True, comment='总股本（万股）'),
        sa.Column('float_share', sa.Float(), nullable=True, comment='流通股本（万股）'),
        sa.Column('total_asset', sa.Float(), nullable=True, comment='总资产（万元）'),
        sa.Column('net_asset', sa.Float(), nullable=True, comment='净资产（万元）'),
        sa.Column('esp', sa.Float(), nullable=True, comment='每股收益'),
        sa.Column('bps', sa.Float(), nullable=True, comment='每股净资产'),
        sa.Column('pe', sa.Float(), nullable=True, comment='市盈率'),
        sa.Column('pb', sa.Float(), nullable=True, comment='市净率'),
        sa.Column('is_st', sa.Boolean(), nullable=False, server_default='0', comment='是否ST股'),
        sa.Column('status', sa.String(length=20), nullable=False, server_default='normal', comment='状态：normal, delisted, suspended'),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP'), comment='创建时间'),
        sa.Column('updated_at', sa.DateTime(), nullable=True, comment='更新时间'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_stock_basics_stock_code', 'stock_basics', ['stock_code'], unique=True)


def downgrade():
    op.drop_index('ix_stock_basics_stock_code', table_name='stock_basics')
    op.drop_table('stock_basics')