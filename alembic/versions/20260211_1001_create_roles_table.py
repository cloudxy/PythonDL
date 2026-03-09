"""create roles table

Revision ID: 20260211_1001_create_roles_table
Revises: 20260211_1000_create_users_table
Create Date: 2026-02-11 10:01:00.000000

"""
from alembic import op
import sqlalchemy as sa

revision = '20260211_1001_create_roles_table'
down_revision = '20260211_1000_create_users_table'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'roles',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False, comment='角色ID'),
        sa.Column('role_name', sa.String(length=50), nullable=False, comment='角色名称'),
        sa.Column('description', sa.String(length=255), nullable=False, comment='角色描述'),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP'), comment='创建时间'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_roles_role_name', 'roles', ['role_name'], unique=True)


def downgrade():
    op.drop_index('ix_roles_role_name', table_name='roles')
    op.drop_table('roles')