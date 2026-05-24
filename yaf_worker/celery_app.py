"""
Celery application configuration.

Run worker with: celery -A yaf_worker.celery_app worker --loglevel=info
"""

from __future__ import annotations

from celery import Celery

celery_app = Celery("yaf")

celery_app.config_from_object("yaf_worker.celery_config")
celery_app.autodiscover_tasks(["yaf_worker.tasks"])
