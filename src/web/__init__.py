"""
Web Dashboard Package.

Provides FastAPI-based web interface for paper trading monitoring.
"""

from .state import DashboardState
from .server import create_app, run_server

__all__ = ["DashboardState", "create_app", "run_server"]
