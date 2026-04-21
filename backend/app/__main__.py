"""Entry point for running the application via `python -m app` or directly via uvicorn."""

import sys
import uvicorn
from app.core.logging import get_log_config

reload = "--reload" in sys.argv

uvicorn.run(
    "app.main:app",
    host="0.0.0.0",
    port=8000,
    reload=reload,
    log_config=get_log_config(),
)
