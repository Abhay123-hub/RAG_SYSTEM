import json
import logging
import logging.config
import time
from pathlib import Path
from typing import Any

import yaml

from config.settings import get_settings
from src.core.context import request_id_var, session_id_var


class JsonFormatter(logging.Formatter):
    """Serialize log records with request and session correlation fields."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "time": self.formatTime(record, "%Y-%m-%dT%H:%M:%S%z"),
            "level": record.levelname,
            "request_id": request_id_var.get(),
            "session_id": session_id_var.get(),
            "module": record.name,
            "event": getattr(record, "event", record.getMessage()),
        }
        for key in ("duration_ms", "data"):
            if hasattr(record, key):
                payload[key] = getattr(record, key)
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging() -> None:
    settings = get_settings()
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    config_path = Path(__file__).parents[2] / "config" / "logging_config.yaml"
    with config_path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    config["root"]["level"] = settings.log_level.upper()
    config["handlers"]["app_file"]["filename"] = str(settings.log_dir / "app.log")
    config["handlers"]["error_file"]["filename"] = str(settings.log_dir / "error.log")
    logging.config.dictConfig(config)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)


def log_event(logger: logging.Logger, event: str, **fields: Any) -> None:
    logger.info(event, extra={"event": event, **fields})


def timed_ms(start: float) -> int:
    return round((time.perf_counter() - start) * 1000)