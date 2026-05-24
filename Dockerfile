FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml ./
RUN pip install --no-cache-dir uv && uv pip install --system -e ".[dev]"
COPY . .
EXPOSE 8000
CMD ["uvicorn", "yaf_api.main:app", "--host", "0.0.0.0", "--port", "8000"]
