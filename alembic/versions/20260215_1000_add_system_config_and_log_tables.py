"""Add system config and log tables

Revision ID: 20260215_1000
Revises: 20260211_1010
Create Date: 2026-02-15 10:00:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '20260215_1000'
down_revision = '20260211_1010'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create system_configs table
    op.create_table('system_configs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('key', sa.String(length=255), nullable=False),
        sa.Column('value', sa.Text(), nullable=False),
        sa.Column('description', sa.String(length=500), nullable=True),
        sa.Column('category', sa.String(length=100), nullable=True),
        sa.Column('is_secret', sa.Boolean(), nullable=True, default=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_system_configs_category'), 'system_configs', ['category'], unique=False)
    op.create_index(op.f('ix_system_configs_id'), 'system_configs', ['id'], unique=False)
    op.create_index(op.f('ix_system_configs_key'), 'system_configs', ['key'], unique=True)
    
    # Create logs table
    op.create_table('logs',
        sa.Column('id', sa.Integer(), nullable=False),
        sa.Column('level', sa.String(length=20), nullable=False),
        sa.Column('module', sa.String(length=100), nullable=True),
        sa.Column('action', sa.String(length=100), nullable=True),
        sa.Column('message', sa.Text(), nullable=False),
        sa.Column('user_id', sa.Integer(), nullable=True),
        sa.Column('ip_address', sa.String(length=50), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=True),
        sa.ForeignKeyConstraint(['user_id'], ['users.id'], ),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index(op.f('ix_logs_action'), 'logs', ['action'], unique=False)
    op.create_index(op.f('ix_logs_created_at'), 'logs', ['created_at'], unique=False)
    op.create_index(op.f('ix_logs_id'), 'logs', ['id'], unique=False)
    op.create_index(op.f('ix_logs_ip_address'), 'logs', ['ip_address'], unique=False)
    op.create_index(op.f('ix_logs_level'), 'logs', ['level'], unique=False)
    op.create_index(op.f('ix_logs_module'), 'logs', ['module'], unique=False)
    op.create_index(op.f('ix_logs_user_id'), 'logs', ['user_id'], unique=False)


def downgrade() -> None:
    # Drop logs table
    op.drop_index(op.f('ix_logs_user_id'), table_name='logs')
    op.drop_index(op.f('ix_logs_module'), table_name='logs')
    op.drop_index(op.f('ix_logs_level'), table_name='logs')
    op.drop_index(op.f('ix_logs_ip_address'), table_name='logs')
    op.drop_index(op.f('ix_logs_id'), table_name='logs')
    op.drop_index(op.f('ix_logs_created_at'), table_name='logs')
    op.drop_index(op.f('ix_logs_action'), table_name='logs')
    op.drop_table('logs')
    
    # Drop system_configs table
    op.drop_index(op.f('ix_system_configs_key'), table_name='system_configs')
    op.drop_index(op.f('ix_system_configs_id'), table_name='system_configs')
    op.drop_index(op.f('ix_system_configs_category'), table_name='system_configs')
    op.drop_table('system_configs')
