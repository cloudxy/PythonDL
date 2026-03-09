"""create face_readings table

Revision ID: 20260211_1007_create_face_readings_table
Revises: 20260211_1006_create_weather_table
Create Date: 2026-02-11 10:07:00.000000

"""
from alembic import op
import sqlalchemy as sa

revision = '20260211_1007_create_face_readings_table'
down_revision = '20260211_1006_create_weather_table'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'face_readings',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False, comment='ID'),
        sa.Column('face_id', sa.String(length=50), nullable=False, comment='面相ID'),
        sa.Column('name', sa.String(length=100), nullable=False, comment='姓名'),
        sa.Column('gender', sa.String(length=10), nullable=False, comment='性别'),
        sa.Column('age', sa.Integer(), nullable=False, comment='年龄'),
        sa.Column('face_shape', sa.String(length=50), nullable=False, comment='脸型'),
        sa.Column('forehead_type', sa.String(length=50), nullable=True, comment='额头类型'),
        sa.Column('eyebrow_type', sa.String(length=50), nullable=True, comment='眉毛类型'),
        sa.Column('eye_type', sa.String(length=50), nullable=True, comment='眼睛类型'),
        sa.Column('nose_type', sa.String(length=50), nullable=True, comment='鼻子类型'),
        sa.Column('mouth_type', sa.String(length=50), nullable=True, comment='嘴巴类型'),
        sa.Column('chin_type', sa.String(length=50), nullable=True, comment='下巴类型'),
        sa.Column('skin_type', sa.String(length=50), nullable=True, comment='皮肤类型'),
        sa.Column('personality_analysis', sa.Text(), nullable=True, comment='性格分析'),
        sa.Column('career_analysis', sa.Text(), nullable=True, comment='事业分析'),
        sa.Column('relationship_analysis', sa.Text(), nullable=True, comment='感情分析'),
        sa.Column('health_analysis', sa.Text(), nullable=True, comment='健康分析'),
        sa.Column('wealth_analysis', sa.Text(), nullable=True, comment='财富分析'),
        sa.Column('luck_score', sa.Float(), nullable=True, comment='运势评分'),
        sa.Column('prediction_result', sa.Text(), nullable=True, comment='预测结果'),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP'), comment='创建时间'),
        sa.Column('updated_at', sa.DateTime(), nullable=True, comment='更新时间'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_face_readings_face_id', 'face_readings', ['face_id'], unique=True)


def downgrade():
    op.drop_index('ix_face_readings_face_id', table_name='face_readings')
    op.drop_table('face_readings')