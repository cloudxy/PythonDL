"""create weather table

Revision ID: create_weather
Revises: create_weather_stations
Create Date: 2024-01-01 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from datetime import datetime


revision = 'create_weather'
down_revision = 'create_weather_stations'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'weather',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False, comment='主键ID'),
        sa.Column('station_code', sa.String(length=20), nullable=False, comment='气象站代码'),
        sa.Column('station_name', sa.String(length=100), nullable=False, comment='气象站名称'),
        sa.Column('observation_time', sa.DateTime(), nullable=False, comment='观测时间'),
        sa.Column('temperature', sa.Float(), nullable=True, comment='气温(摄氏度)'),
        sa.Column('humidity', sa.Float(), nullable=True, comment='相对湿度(%)'),
        sa.Column('pressure', sa.Float(), nullable=True, comment='气压(hPa)'),
        sa.Column('wind_speed', sa.Float(), nullable=True, comment='风速(m/s)'),
        sa.Column('wind_direction', sa.Integer(), nullable=True, comment='风向(度)'),
        sa.Column('precipitation', sa.Float(), nullable=True, comment='降水量(mm)'),
        sa.Column('visibility', sa.Float(), nullable=True, comment='能见度(km)'),
        sa.Column('cloud_cover', sa.Float(), nullable=True, comment='云量(%)'),
        sa.Column('weather_condition', sa.String(length=50), nullable=True, comment='天气状况'),
        sa.Column('weather_code', sa.String(length=10), nullable=True, comment='天气代码'),
        sa.Column('dew_point', sa.Float(), nullable=True, comment='露点温度(摄氏度)'),
        sa.Column('feels_like', sa.Float(), nullable=True, comment='体感温度(摄氏度)'),
        sa.Column('uv_index', sa.Float(), nullable=True, comment='紫外线指数'),
        sa.Column('created_at', sa.DateTime(), default=datetime.now, nullable=True, comment='创建时间'),
        sa.Column('updated_at', sa.DateTime(), default=datetime.now, onupdate=datetime.now, nullable=True, comment='更新时间'),
        sa.PrimaryKeyConstraint('id'),
        comment='气象数据表'
    )
    op.create_index('idx_weather_station_time', 'weather', ['station_code', 'observation_time'])
    op.create_index('idx_weather_observation_time', 'weather', ['observation_time'])
    op.create_index('ix_weather_station_code', 'weather', ['station_code'])


def downgrade():
    op.drop_index('ix_weather_station_code', table_name='weather')
    op.drop_index('idx_weather_observation_time', table_name='weather')
    op.drop_index('idx_weather_station_time', table_name='weather')
    op.drop_table('weather')
