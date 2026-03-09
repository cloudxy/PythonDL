"""create role_permissions table

Revision ID: 20260211_1003_create_role_permissions_table
Revises: 20260211_1002_create_permissions_table
Create Date: 2026-02-11 10:03:00.000000

"""
from alembic import op
import sqlalchemy as sa

revision = '20260211_1003_create_role_permissions_table'
down_revision = '20260211_1002_create_permissions_table'
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        'role_permissions',
        sa.Column('id', sa.Integer(), autoincrement=True, nullable=False, comment='ID'),
        sa.Column('role_id', sa.Integer(), nullable=False, comment='角色ID'),
        sa.Column('permission_id', sa.Integer(), nullable=False, comment='权限ID'),
        sa.ForeignKeyConstraint(['permission_id'], ['permissions.id'], ondelete='CASCADE'),
        sa.ForeignKeyConstraint(['role_id'], ['roles.id'], ondelete='CASCADE'),
        sa.PrimaryKeyConstraint('id')
    )


def downgrade():
    op.drop_table('role_permissions')