"""
Designs router — CRUD for antenna designs.

POST   /api/v1/designs       Create a new design
GET    /api/v1/designs       List designs
GET    /api/v1/designs/{id}  Get design detail
DELETE /api/v1/designs/{id}  Delete a design
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from yaf_core.domain.design import (
    BoundingBox,
    Design,
    DesignSpec,
    DesignState,
    PatternTarget,
    Polarization,
)

router = APIRouter(prefix="/api/v1/designs", tags=["designs"])

# In-memory store (replace with DB in production)
_designs: dict[uuid.UUID, Design] = {}


@router.post("", response_model=Design)
async def create_design(spec: DesignSpec) -> Design:
    """Create a new antenna design."""
    design = Design(spec=spec, state=DesignState.DRAFT)
    _designs[design.id] = design
    return design


@router.get("", response_model=list[Design])
async def list_designs() -> list[Design]:
    """List all designs."""
    return list(_designs.values())


@router.get("/{design_id}", response_model=Design)
async def get_design(design_id: uuid.UUID) -> Design:
    """Get a design by ID."""
    if design_id not in _designs:
        raise HTTPException(status_code=404, detail="Design not found")
    return _designs[design_id]


@router.delete("/{design_id}")
async def delete_design(design_id: uuid.UUID) -> dict[str, str]:
    """Delete a design."""
    if design_id not in _designs:
        raise HTTPException(status_code=404, detail="Design not found")
    del _designs[design_id]
    return {"status": "deleted"}
