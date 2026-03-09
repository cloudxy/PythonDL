"""create consumption_categories table

Revision ID: create_consumption_categories
Revises: create_consumption
Create Date: 2024-01-01 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from datetime import datetime


revision = 'create_consumption_categories'
down_revision = 'create_consumption'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'consumption_categories',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False, comment='主键ID'),
        sa.Column('category_code', sa.String(length=20), nullable=False, comment='类别代码'),
        sa.Column('category_name', sa.String(length=100), nullable=False, comment='类别名称'),
        sa.Column('parent_code', sa.String(length=20), nullable=True, comment='父类别代码'),
        sa.Column('level', sa.Integer(), nullable=True, comment='类别层级'),
        sa.Column('description', sa.Text(), nullable=True, comment='类别描述'),
        sa.Column('weight', sa.Float(), nullable=True, comment='权重'),
        sa.Column('is_active', sa.Integer(), default=1, nullable=True, comment='是否启用 0-禁用 1-启用'),
        sa.Column('created_at', sa.DateTime(), default=datetime.now, nullable=True, comment='创建时间'),
        sa.Column('updated_at', sa.DateTime(), default=datetime.now, onupdate=datetime.now, nullable=True, comment='更新时间'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('category_code'),
        comment='消费类别表'
    )
    op.create_index('idx_consumption_category_code', 'consumption_categories', ['category_code'])
    op.create_index('idx_consumption_category_parent', 'consumption_categories', ['parent_code'])


def downgrade():
    op.drop_index('idx_consumption_category_parent', table_name='consumption_categories')
    op.drop_index('idx_consumption_category_code', table_name='consumption_categories')
    op.drop_table('consumption_categories')
