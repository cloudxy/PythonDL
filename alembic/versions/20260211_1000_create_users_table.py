"""create users table

Revision ID: 20260211_1000_create_users_table
Revises: 
Create Date: 2026-02-11 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa

revision = '20260211_1000_create_users_table'
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'users',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False, comment='用户ID'),
        sa.Column('username', sa.String(length=50), nullable=False, comment='用户名'),
        sa.Column('password', sa.String(length=100), nullable=False, comment='密码'),
        sa.Column('email', sa.String(length=100), nullable=False, comment='邮箱'),
        sa.Column('role', sa.String(length=20), nullable=False, server_default='user', comment='角色'),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP'), comment='创建时间'),
        sa.Column('last_login', sa.DateTime(), nullable=True, comment='最后登录时间'),
        sa.Column('is_active', sa.Boolean(), nullable=False, server_default='1', comment='是否激活'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_users_username', 'users', ['username'], unique=True)
    op.create_index('ix_users_email', 'users', ['email'])


def downgrade():
    op.drop_index('ix_users_email', table_name='users')
    op.drop_index('ix_users_username', table_name='users')
    op.drop_table('users')