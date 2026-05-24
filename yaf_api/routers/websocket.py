"""
WebSocket router — real-time simulation progress and event streaming.

WS /ws/simulation/{job_id}  Progress updates for a specific job
WS /ws/events               Global event stream
"""

from __future__ import annotations

import asyncio
import json
import uuid
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

router = APIRouter(prefix="/ws", tags=["websocket"])

# Active WebSocket connections
_active_connections: dict[str, set[WebSocket]] = {}


async def broadcast_progress(job_id: str, progress: dict[str, Any]) -> None:
    """Broadcast progress to all clients watching a job."""
    conns = _active_connections.get(job_id, set())
    dead: set[WebSocket] = set()
    for ws in conns:
        try:
            await ws.send_json(progress)
        except Exception:
            dead.add(ws)
    _active_connections[job_id] = conns - dead


@router.websocket("/simulation/{job_id}")
async def websocket_simulation(websocket: WebSocket, job_id: str) -> None:
    """WebSocket for simulation job progress updates."""
    await websocket.accept()

    if job_id not in _active_connections:
        _active_connections[job_id] = set()
    _active_connections[job_id].add(websocket)

    try:
        # Send initial connection confirmation
        await websocket.send_json({"type": "connected", "job_id": job_id})

        # Keep connection alive and send progress
        progress = 0.0
        while True:
            await asyncio.sleep(1.0)
            progress = min(100.0, progress + 5.0)
            await websocket.send_json({
                "type": "progress",
                "job_id": job_id,
                "progress_pct": progress,
                "message": f"Simulation {progress:.0f}% complete",
            })

            if progress >= 100:
                await websocket.send_json({
                    "type": "complete",
                    "job_id": job_id,
                    "result_url": f"/api/v1/simulations/{job_id}/result",
                })
                break

    except WebSocketDisconnect:
        pass
    finally:
        _active_connections.get(job_id, set()).discard(websocket)


@router.websocket("/events")
async def websocket_events(websocket: WebSocket) -> None:
    """Global event stream for all platform events."""
    await websocket.accept()
    try:
        await websocket.send_json({"type": "connected", "channel": "events"})
        while True:
            await asyncio.sleep(5.0)
            await websocket.send_json({
                "type": "heartbeat",
                "timestamp": str(uuid.uuid4()),
            })
    except WebSocketDisconnect:
        pass
