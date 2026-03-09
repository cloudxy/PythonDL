"""create permissions table

Revision ID: 20260211_1002_create_permissions_table
Revises: 20260211_1001_create_roles_table
Create Date: 2026-02-11 10:02:00.000000

"""
from alembic import op
import sqlalchemy as sa

revision = '20260211_1002_create_permissions_table'
down_revision = '20260211_1001_create_roles_table'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'permissions',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False, comment='权限ID'),
        sa.Column('permission_name', sa.String(length=50), nullable=False, comment='权限名称'),
        sa.Column('description', sa.String(length=255), nullable=False, comment='权限描述'),
        sa.Column('created_at', sa.DateTime(), nullable=False, server_default=sa.text('CURRENT_TIMESTAMP'), comment='创建时间'),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('ix_permissions_permission_name', 'permissions', ['permission_name'], unique=True)


def downgrade():
    op.drop_index('ix_permissions_permission_name', table_name='permissions')
    op.drop_table('permissions')