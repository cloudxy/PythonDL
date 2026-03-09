"""create weather table

Revision ID: 20260211_1006_create_weather_table
Revises: 20260211_1005_create_stocks_table
Create Date: 2026-02-11 10:06:00.000000

"""
from alembic import op
import sqlalchemy as sa

revision = '20260211_1006_create_weather_table'
down_revision = '20260211_1005_create_stocks_table'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'weather',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False, comment='ID'),
        sa.Column('city', sa.String(length=100), nullable=False, comment='城市名称'),
        sa.Column('district', sa.String(length=100), nullable=True, comment='区域/区县'),
        sa.Column('observation_time', sa.DateTime(), nullable=False, comment='观测时间'),
        sa.Column('temperature', sa.Float(), nullable=True, comment='温度(℃)'),
        sa.Column('humidity', sa.Float(), nullable=True, comment='相对湿度(%)'),
        sa.Column('pressure', sa.Float(), nullable=True, comment='气压(hPa)'),
        sa.Column('wind_direction', sa.String(length=20), nullable=True, comment='风向'),
        sa.Column('wind_speed', sa.Float(), nullable=True, comment='风速(m/s)'),
        sa.Column('wind_scale', sa.Integer(), nullable=True, comment='风力等级'),
        sa.Column('visibility', sa.Float(), nullable=True, comment='能见度(km)'),
        sa.Column('precipitation', sa.Float(), nullable=True, comment='降水量(mm)'),
        sa.Column('weather_condition', sa.String(length=50), nullable=True, comment='天气现象'),
        sa.Column('cloud_cover', sa.Integer(), nullable=True, comment='云量(%)'),
        sa.Column('dew_point', sa.Float(), nullable=True, comment='露点温度(℃)'),
        sa.Column('uv_index', sa.Float(), nullable=True, comment='紫外线指数'),
        sa.Column('aqi', sa.Integer(), nullable=True, comment='空气质量指数'),
        sa.Column('pm25', sa.Float(), nullable=True, comment='PM2.5浓度(μg/m³)'),
        sa.Column('pm10', sa.Float(), nullable=True, comment='PM10浓度(μg/m³)'),
        sa.Column('so2', sa.Float(), nullable=True, comment='SO2浓度(μg/m³)'),
        sa.Column('no2', sa.Float(), nullable=True, comment='NO2浓度(μg/m³)'),
        sa.Column('co', sa.Float(), nullable=True, comment='CO浓度(mg/m³)'),
        sa.Column('o3', sa.Float(), nullable=True, comment='O3浓度(μg/m³)'),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP'), comment='创建时间'),
        sa.Column('updated_at', sa.DateTime(), nullable=True, comment='更新时间'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_weather_city', 'weather', ['city'])
    op.create_index('ix_weather_observation_time', 'weather', ['observation_time'])


def downgrade():
    op.drop_index('ix_weather_observation_time', table_name='weather')
    op.drop_index('ix_weather_city', table_name='weather')
    op.drop_table('weather')