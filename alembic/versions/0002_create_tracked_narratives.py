"""Create tracked_narratives table for global monitoring and alerts.

Revision ID: 0002
Revises: 0001
Create Date: 2026-11-25 17:45:00
"""

from __future__ import annotations

from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

revision: str = "0002"
down_revision: str | None = "0001"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Create tracked_narratives table."""
    op.create_table(
        "tracked_narratives",
        sa.Column("id", sa.String(length=64), nullable=False),
        sa.Column("name", sa.String(length=160), nullable=False),
        sa.Column("query", sa.String(length=256), nullable=False),
        sa.Column("sources", sa.JSON(), nullable=False),
        sa.Column("active", sa.Boolean(), nullable=False),
        sa.Column("alert_velocity_threshold", sa.Float(), nullable=False),
        sa.Column("alert_rsi_threshold", sa.Float(), nullable=False),
        sa.Column("last_checked_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_infected_count", sa.Integer(), nullable=False),
        sa.Column("last_skeptic_count", sa.Integer(), nullable=False),
        sa.Column("last_max_velocity", sa.Float(), nullable=True),
        sa.Column("last_inflection_time", sa.Float(), nullable=True),
        sa.Column("last_rsi", sa.Float(), nullable=True),
        sa.Column("alert_triggered", sa.Boolean(), nullable=False),
        sa.Column("alert_message", sa.Text(), nullable=True),
        sa.Column("country_counts", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_tracked_narratives_active"),
        "tracked_narratives",
        ["active"],
        unique=False,
    )


def downgrade() -> None:
    """Drop tracked_narratives table."""
    op.drop_index(op.f("ix_tracked_narratives_active"), table_name="tracked_narratives")
    op.drop_table("tracked_narratives")
