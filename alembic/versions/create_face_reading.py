"""create face_reading table

Revision ID: create_face_reading
Revises: create_bazi
Create Date: 2024-01-01 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from datetime import datetime


revision = 'create_face_reading'
down_revision = 'create_bazi'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'face_reading',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False, comment='主键ID'),
        sa.Column('user_id', sa.String(length=50), nullable=False, comment='用户ID'),
        sa.Column('session_id', sa.String(length=100), nullable=False, comment='会话ID'),
        sa.Column('face_shape', sa.String(length=50), nullable=True, comment='脸型'),
        sa.Column('forehead', sa.String(length=50), nullable=True, comment='额头'),
        sa.Column('eyebrows', sa.String(length=50), nullable=True, comment='眉毛'),
        sa.Column('eyes', sa.String(length=50), nullable=True, comment='眼睛'),
        sa.Column('nose', sa.String(length=50), nullable=True, comment='鼻子'),
        sa.Column('mouth', sa.String(length=50), nullable=True, comment='嘴巴'),
        sa.Column('ears', sa.String(length=50), nullable=True, comment='耳朵'),
        sa.Column('chin', sa.String(length=50), nullable=True, comment='下巴'),
        sa.Column('facial_features', sa.JSON(), nullable=True, comment='面部特征'),
        sa.Column('personality_traits', sa.JSON(), nullable=True, comment='性格特征'),
        sa.Column('career_potential', sa.JSON(), nullable=True, comment='事业潜力'),
        sa.Column('wealth_potential', sa.JSON(), nullable=True, comment='财富潜力'),
        sa.Column('love_potential', sa.JSON(), nullable=True, comment='爱情潜力'),
        sa.Column('health_indicators', sa.JSON(), nullable=True, comment='健康指标'),
        sa.Column('overall_score', sa.Float(), nullable=True, comment='综合评分'),
        sa.Column('analysis_text', sa.Text(), nullable=True, comment='分析文本'),
        sa.Column('advice_text', sa.Text(), nullable=True, comment='建议文本'),
        sa.Column('created_at', sa.DateTime(), default=datetime.now, nullable=True, comment='创建时间'),
        sa.Column('updated_at', sa.DateTime(), default=datetime.now, onupdate=datetime.now, nullable=True, comment='更新时间'),
        sa.PrimaryKeyConstraint('id'),
        comment='面相分析数据表'
    )
    op.create_index('idx_face_user_time', 'face_reading', ['user_id', 'created_at'])
    op.create_index('ix_face_reading_user_id', 'face_reading', ['user_id'])
    op.create_index('ix_face_reading_session_id', 'face_reading', ['session_id'])


def downgrade():
    op.drop_index('ix_face_reading_session_id', table_name='face_reading')
    op.drop_index('ix_face_reading_user_id', table_name='face_reading')
    op.drop_index('idx_face_user_time', table_name='face_reading')
    op.drop_table('face_reading')
