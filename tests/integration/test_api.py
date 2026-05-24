"""Integration test: FastAPI health endpoint."""

import pytest
from httpx import AsyncClient, ASGITransport

from yaf_api.main import app


@pytest.mark.asyncio
async def test_health():
    """Verify the health endpoint returns 200 OK."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["version"] == "0.1.0"

@pytest.mark.asyncio
async def test_api_v1_health():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/v1/health")
        assert resp.status_code == 200
