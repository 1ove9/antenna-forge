"""
YAF Database Models — SQLAlchemy 2.0 ORM models.

Maps domain entities to PostgreSQL tables.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class DesignModel(Base):
    __tablename__ = "designs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    name: Mapped[str] = mapped_column(String(255))
    spec_json: Mapped[dict] = mapped_column(JSONB)
    state: Mapped[str] = mapped_column(String(50), default="draft")
    current_version: Mapped[int] = mapped_column(Integer, default=0)
    tags: Mapped[list] = mapped_column(JSONB, default=list)
    parent_design_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc),
        onupdate=lambda: datetime.now(timezone.utc),
    )


class SimulationJobModel(Base):
    __tablename__ = "simulation_jobs"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    design_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("designs.id"))
    design_version: Mapped[int] = mapped_column(Integer)
    solver_name: Mapped[str] = mapped_column(String(50))
    spec_json: Mapped[dict] = mapped_column(JSONB)
    state: Mapped[str] = mapped_column(String(50), default="pending")
    priority: Mapped[int] = mapped_column(Integer, default=0)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=lambda: datetime.now(timezone.utc)
    )
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class SimulationResultModel(Base):
    __tablename__ = "simulation_results"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    job_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), ForeignKey("simulation_jobs.id"), unique=True
    )
    solver_name: Mapped[str] = mapped_column(String(50))
    solver_version: Mapped[str] = mapped_column(String(20))
    status: Mapped[str] = mapped_column(String(20))
    gain_dbi: Mapped[float | None] = mapped_column(Float, nullable=True)
    efficiency: Mapped[float | None] = mapped_column(Float, nullable=True)
    vswr: Mapped[float | None] = mapped_column(Float, nullable=True)
    bandwidth_hz: Mapped[float | None] = mapped_column(Float, nullable=True)
    result_json: Mapped[dict] = mapped_column(JSONB, default=dict)
    simulation_time_sec: Mapped[float] = mapped_column(Float, default=0.0)
