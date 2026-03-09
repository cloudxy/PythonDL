"""create weather_stations table

Revision ID: create_weather_stations
Revises: 
Create Date: 2024-01-01 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from datetime import datetime


revision = 'create_weather_stations'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'weather_stations',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False, comment='主键ID'),
        sa.Column('station_code', sa.String(length=20), nullable=False, comment='气象站代码'),
        sa.Column('station_name', sa.String(length=100), nullable=False, comment='气象站名称'),
        sa.Column('country', sa.String(length=50), nullable=True, comment='国家'),
        sa.Column('province', sa.String(length=50), nullable=True, comment='省份'),
        sa.Column('city', sa.String(length=50), nullable=True, comment='城市'),
        sa.Column('district', sa.String(length=50), nullable=True, comment='区县'),
        sa.Column('latitude', sa.Float(), nullable=True, comment='纬度'),
        sa.Column('longitude', sa.Float(), nullable=True, comment='经度'),
        sa.Column('altitude', sa.Float(), nullable=True, comment='海拔高度(米)'),
        sa.Column('station_type', sa.String(length=20), nullable=True, comment='气象站类型'),
        sa.Column('data_source', sa.String(length=50), nullable=True, comment='数据来源'),
        sa.Column('is_active', sa.Integer(), default=1, nullable=True, comment='是否启用 0-禁用 1-启用'),
        sa.Column('created_at', sa.DateTime(), default=datetime.now, nullable=True, comment='创建时间'),
        sa.Column('updated_at', sa.DateTime(), default=datetime.now, onupdate=datetime.now, nullable=True, comment='更新时间'),
        sa.PrimaryKeyConstraint('id'),
        sa.UniqueConstraint('station_code'),
        comment='气象站信息表'
    )
    op.create_index('idx_weather_station_code', 'weather_stations', ['station_code'])
    op.create_index('idx_weather_station_city', 'weather_stations', ['city'])


def downgrade():
    op.drop_index('idx_weather_station_city', table_name='weather_stations')
    op.drop_index('idx_weather_station_code', table_name='weather_stations')
    op.drop_table('weather_stations')
