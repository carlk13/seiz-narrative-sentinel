"""Create simulation_runs and trajectory_data tables.

Revision ID: 0001
Revises:
Create Date: 2026-11-17 19:55:00
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0001"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create simulation_runs and trajectory_data tables."""
    op.create_table(
        "simulation_runs",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("task_type", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("initial_state", sa.JSON(), nullable=True),
        sa.Column("parameters", sa.JSON(), nullable=True),
        sa.Column("sse", sa.Float(), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
        sa.Column("error_type", sa.String(length=128), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_simulation_runs_status"),
        "simulation_runs",
        ["status"],
        unique=False,
    )

    op.create_table(
        "trajectory_data",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("simulation_run_id", sa.String(length=64), nullable=False),
        sa.Column("t", sa.JSON(), nullable=False),
        sa.Column("S", sa.JSON(), nullable=False),
        sa.Column("E", sa.JSON(), nullable=False),
        sa.Column("I", sa.JSON(), nullable=False),
        sa.Column("Z", sa.JSON(), nullable=False),
        sa.Column("N", sa.JSON(), nullable=False),
        sa.Column("percentiles", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["simulation_run_id"],
            ["simulation_runs.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("simulation_run_id"),
    )
    op.create_index(
        op.f("ix_trajectory_data_simulation_run_id"),
        "trajectory_data",
        ["simulation_run_id"],
        unique=True,
    )


def downgrade() -> None:
    """Drop trajectory_data and simulation_runs tables."""
    op.drop_index(op.f("ix_trajectory_data_simulation_run_id"), table_name="trajectory_data")
    op.drop_table("trajectory_data")
    op.drop_index(op.f("ix_simulation_runs_status"), table_name="simulation_runs")
    op.drop_table("simulation_runs")
