"""create fortune_tellings table

Revision ID: 20260211_1008_create_fortune_tellings_table
Revises: 20260211_1007_create_face_readings_table
Create Date: 2026-02-11 10:08:00.000000

"""
from alembic import op
import sqlalchemy as sa

revision = '20260211_1008_create_fortune_tellings_table'
down_revision = '20260211_1007_create_face_readings_table'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'fortune_tellings',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False, comment='ID'),
        sa.Column('fortune_id', sa.String(length=50), nullable=False, comment='命理ID'),
        sa.Column('name', sa.String(length=100), nullable=False, comment='姓名'),
        sa.Column('gender', sa.String(length=10), nullable=False, comment='性别'),
        sa.Column('birth_date', sa.DateTime(), nullable=False, comment='出生日期'),
        sa.Column('birth_time', sa.String(length=20), nullable=True, comment='出生时间'),
        sa.Column('birth_place', sa.String(length=100), nullable=True, comment='出生地点'),
        sa.Column('zodiac', sa.String(length=20), nullable=True, comment='生肖'),
        sa.Column('constellation', sa.String(length=20), nullable=True, comment='星座'),
        sa.Column('five_elements', sa.String(length=100), nullable=True, comment='五行分析'),
        sa.Column('bazi', sa.String(length=200), nullable=True, comment='八字分析'),
        sa.Column('overall_luck', sa.Text(), nullable=True, comment='综合运势'),
        sa.Column('career_luck', sa.Text(), nullable=True, comment='事业运势'),
        sa.Column('love_luck', sa.Text(), nullable=True, comment='爱情运势'),
        sa.Column('wealth_luck', sa.Text(), nullable=True, comment='财富运势'),
        sa.Column('health_luck', sa.Text(), nullable=True, comment='健康运势'),
        sa.Column('luck_score', sa.Float(), nullable=True, comment='运势评分'),
        sa.Column('lucky_direction', sa.String(length=100), nullable=True, comment='吉祥方位'),
        sa.Column('lucky_colors', sa.String(length=100), nullable=True, comment='吉祥颜色'),
        sa.Column('lucky_numbers', sa.String(length=100), nullable=True, comment='吉祥数字'),
        sa.Column('prediction_result', sa.Text(), nullable=True, comment='预测结果'),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP'), comment='创建时间'),
        sa.Column('updated_at', sa.DateTime(), nullable=True, comment='更新时间'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_fortune_tellings_fortune_id', 'fortune_tellings', ['fortune_id'], unique=True)


def downgrade():
    op.drop_index('ix_fortune_tellings_fortune_id', table_name='fortune_tellings')
    op.drop_table('fortune_tellings')