"""create weather_forecasts table

Revision ID: create_weather_forecasts
Revises: create_weather
Create Date: 2024-01-01 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from datetime import datetime


revision = 'create_weather_forecasts'
down_revision = 'create_weather'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'weather_forecasts',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False, comment='主键ID'),
        sa.Column('station_code', sa.String(length=20), nullable=False, comment='气象站代码'),
        sa.Column('forecast_time', sa.DateTime(), nullable=False, comment='预报时间'),
        sa.Column('valid_time', sa.DateTime(), nullable=False, comment='预报有效时间'),
        sa.Column('forecast_type', sa.String(length=20), nullable=True, comment='预报类型'),
        sa.Column('temperature_max', sa.Float(), nullable=True, comment='最高温度(摄氏度)'),
        sa.Column('temperature_min', sa.Float(), nullable=True, comment='最低温度(摄氏度)'),
        sa.Column('temperature_avg', sa.Float(), nullable=True, comment='平均温度(摄氏度)'),
        sa.Column('humidity', sa.Float(), nullable=True, comment='相对湿度(%)'),
        sa.Column('pressure', sa.Float(), nullable=True, comment='气压(hPa)'),
        sa.Column('wind_speed', sa.Float(), nullable=True, comment='风速(m/s)'),
        sa.Column('wind_direction', sa.Integer(), nullable=True, comment='风向(度)'),
        sa.Column('precipitation', sa.Float(), nullable=True, comment='降水量(mm)'),
        sa.Column('precipitation_probability', sa.Float(), nullable=True, comment='降水概率(%)'),
        sa.Column('weather_condition', sa.String(length=50), nullable=True, comment='天气状况'),
        sa.Column('weather_code', sa.String(length=10), nullable=True, comment='天气代码'),
        sa.Column('visibility', sa.Float(), nullable=True, comment='能见度(km)'),
        sa.Column('cloud_cover', sa.Float(), nullable=True, comment='云量(%)'),
        sa.Column('uv_index', sa.Float(), nullable=True, comment='紫外线指数'),
        sa.Column('confidence', sa.Float(), nullable=True, comment='预报置信度(%)'),
        sa.Column('created_at', sa.DateTime(), default=datetime.now, nullable=True, comment='创建时间'),
        sa.Column('updated_at', sa.DateTime(), default=datetime.now, onupdate=datetime.now, nullable=True, comment='更新时间'),
        sa.PrimaryKeyConstraint('id'),
        comment='天气预报数据表'
    )
    op.create_index('idx_weather_forecast_station_time', 'weather_forecasts', ['station_code', 'forecast_time'])
    op.create_index('idx_weather_forecast_valid_time', 'weather_forecasts', ['valid_time'])
    op.create_index('ix_weather_forecasts_station_code', 'weather_forecasts', ['station_code'])


def downgrade():
    op.drop_index('ix_weather_forecasts_station_code', table_name='weather_forecasts')
    op.drop_index('idx_weather_forecast_valid_time', table_name='weather_forecasts')
    op.drop_index('idx_weather_forecast_station_time', table_name='weather_forecasts')
    op.drop_table('weather_forecasts')
