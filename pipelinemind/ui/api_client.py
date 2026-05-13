"""
Centralised API base URL resolver.
Reads API_BASE_URL from the environment.

Local dev:  API_BASE_URL not set -> http://localhost:8000
Docker:     API_BASE_URL=http://api:8000 (set via docker-compose.yml)

All UI components import _API_BASE from here — never hardcode the URL.
"""
from __future__ import annotations

import os

_API_BASE: str = os.environ.get("API_BASE_URL", "http://localhost:8000").rstrip("/")
