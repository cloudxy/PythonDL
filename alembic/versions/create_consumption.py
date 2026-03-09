"""create consumption table

Revision ID: create_consumption
Revises: create_face_reading
Create Date: 2024-01-01 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from datetime import datetime


revision = 'create_consumption'
down_revision = 'create_face_reading'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'consumption',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False, comment='主键ID'),
        sa.Column('region', sa.String(length=50), nullable=False, comment='地区'),
        sa.Column('region_level', sa.String(length=20), nullable=True, comment='地区级别'),
        sa.Column('data_date', sa.DateTime(), nullable=False, comment='数据日期'),
        sa.Column('total_consumption', sa.Float(), nullable=True, comment='总消费额'),
        sa.Column('per_capita_consumption', sa.Float(), nullable=True, comment='人均消费额'),
        sa.Column('food_consumption', sa.Float(), nullable=True, comment='食品消费'),
        sa.Column('clothing_consumption', sa.Float(), nullable=True, comment='衣着消费'),
        sa.Column('housing_consumption', sa.Float(), nullable=True, comment='居住消费'),
        sa.Column('transportation_consumption', sa.Float(), nullable=True, comment='交通消费'),
        sa.Column('education_consumption', sa.Float(), nullable=True, comment='教育消费'),
        sa.Column('medical_consumption', sa.Float(), nullable=True, comment='医疗消费'),
        sa.Column('entertainment_consumption', sa.Float(), nullable=True, comment='娱乐消费'),
        sa.Column('other_consumption', sa.Float(), nullable=True, comment='其他消费'),
        sa.Column('consumption_structure', sa.JSON(), nullable=True, comment='消费结构'),
        sa.Column('year_over_year', sa.Float(), nullable=True, comment='同比增长率'),
        sa.Column('month_over_month', sa.Float(), nullable=True, comment='环比增长率'),
        sa.Column('consumer_confidence_index', sa.Float(), nullable=True, comment='消费者信心指数'),
        sa.Column('consumer_satisfaction_index', sa.Float(), nullable=True, comment='消费者满意度指数'),
        sa.Column('created_at', sa.DateTime(), default=datetime.now, nullable=True, comment='创建时间'),
        sa.Column('updated_at', sa.DateTime(), default=datetime.now, onupdate=datetime.now, nullable=True, comment='更新时间'),
        sa.PrimaryKeyConstraint('id'),
        comment='宏观消费数据表'
    )
    op.create_index('idx_consumption_region_date', 'consumption', ['region', 'data_date'])
    op.create_index('idx_consumption_date', 'consumption', ['data_date'])
    op.create_index('ix_consumption_region', 'consumption', ['region'])


def downgrade():
    op.drop_index('ix_consumption_region', table_name='consumption')
    op.drop_index('idx_consumption_date', table_name='consumption')
    op.drop_index('idx_consumption_region_date', table_name='consumption')
    op.drop_table('consumption')
