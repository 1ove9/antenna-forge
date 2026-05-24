"""
Celery configuration — Redis broker settings.
"""

from __future__ import annotations

import os

broker_url = os.getenv("CELERY_BROKER_URL", "redis://localhost:6379/1")
result_backend = os.getenv("CELERY_RESULT_BACKEND", "redis://localhost:6379/2")

task_serializer = "json"
result_serializer = "json"
accept_content = ["json"]
timezone = "UTC"
enable_utc = True

task_track_started = True
task_acks_late = True
worker_prefetch_multiplier = 1

task_routes = {
    "yaf_worker.tasks.simulate.*": {"queue": "simulations"},
    "yaf_worker.tasks.optimize.*": {"queue": "optimizations"},
    "yaf_worker.tasks.generate.*": {"queue": "generation"},
}
