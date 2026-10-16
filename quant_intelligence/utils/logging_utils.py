"""
Structured logging + decision-ID generation.

Every meaningful decision in the pipeline (market state update, ranking,
setup, risk decision, order, fill) gets logged to both a rotating file log
and the `system_events` table, tagged with a decision_id so it can be
audited end-to-end from the Streamlit UI.
"""
from __future__ import annotations

import json
import logging
import re
import uuid
from logging.handlers import RotatingFileHandler
from pathlib import Path

from quant_intelligence.config.settings import LOGS_DIR

_LOGGER_NAME = "quant_intelligence"


_JWT = re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{5,}")
_SECRET_FIELD = re.compile(
    r'(?i)("?(?:access[-_ ]?token|api[-_ ]?key|secret|password|passwd|totp[-_ ]?secret|authorization|dhan_pin|pin)"?\s*[:=]\s*"?)'
    r'((?:Bearer\s+)?[^",\s}\]]+)'
)


def redact(text: str) -> str:
    """Remove credential-looking content (JWTs, and the value of fields named token / secret / password / PIN ...) from
    text that is about to be logged or shown. A last line of defence: callers should not pass secrets in the first place."""
    text = _JWT.sub("[jwt-redacted]", text)
    return _SECRET_FIELD.sub(lambda m: m.group(1) + "[redacted]", text)


def new_decision_id(prefix: str = "DEC") -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


def get_logger() -> logging.Logger:
    logger = logging.getLogger(_LOGGER_NAME)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)

    fmt = logging.Formatter(
        '{"ts": "%(asctime)s", "level": "%(levelname)s", "component": "%(name)s", "msg": %(message)s}'
    )

    file_handler = RotatingFileHandler(
        Path(LOGS_DIR) / "system.log", maxBytes=5_000_000, backupCount=5
    )
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(fmt)
    logger.addHandler(stream_handler)

    return logger


def log_event(component: str, message: str, level: str = "INFO", **details) -> None:
    """Log a structured event to file/console AND persist to system_events table."""
    logger = get_logger().getChild(component)
    payload = redact(json.dumps({"message": message, "details": details}, default=str))
    getattr(logger, level.lower(), logger.info)(payload)

    try:
        from quant_intelligence.database.db import get_session
        from quant_intelligence.database.models import SystemEvent

        with get_session() as session:
            session.add(
                SystemEvent(
                    component=component,
                    level=level,
                    message=redact(message),
                    details=json.loads(redact(json.dumps(details, default=str))) if details else None,
                )
            )
    except Exception:
        # Logging must never crash the pipeline. DB may not be initialised yet.
        pass
