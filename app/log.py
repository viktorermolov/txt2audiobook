import logging
import os
import sys


def setup_logging(level: str | None = None) -> None:
    lvl = (level or os.environ.get("LOG_LEVEL") or "INFO").upper()
    logging.basicConfig(
        stream=sys.stdout,
        level=lvl,
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # Tame chatty libraries.
    logging.getLogger("aiogram").setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("watchdog").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)
