.PHONY: install dev test lint demo up down clean

install:
	uv pip install --system -e ".[dev]"

dev:
	uvicorn yaf_api.main:app --reload --port 8000

worker:
	celery -A yaf_worker.celery_app worker --loglevel=info

test:
	pytest tests/ -x -v

lint:
	ruff check . && mypy yaf_core yaf_ai yaf_solvers

demo-fdtd:
	python -m yaf_ai.differentiable.diff_fdtd_jax --demo

demo-vae:
	python -m yaf_ai.generative.vae_designer --train --epochs 20

demo-bayesian:
	python -m yaf_ai.optimization.bayesian --demo

demo-pipeline:
	python -m yaf_ai.inverse_design.pipeline --demo

demo-dipole:
	python scripts/demo_dipole.py

up:
	docker compose up -d

down:
	docker compose down

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name '*.pyc' -delete 2>/dev/null || true
