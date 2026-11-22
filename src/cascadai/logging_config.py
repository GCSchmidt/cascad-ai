import logging
import os
from datetime import datetime
from pathlib import Path


LOGGER_NAME = "cascadai"
DEFAULT_MODE = "file"
DEFAULT_LEVEL = "INFO"
VALID_MODES = {"off", "file", "console", "both"}

_configured = False


def _find_project_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


def _resolve_log_dir() -> Path:
    configured_dir = os.getenv("CASCAD_AI_LOG_DIR")
    if configured_dir:
        return Path(configured_dir).expanduser().resolve()
    return _find_project_root() / "logs"


def configure_logging(mode: str | None = None, force: bool = False) -> None:
    global _configured

    if _configured and not force:
        return

    logger = logging.getLogger(LOGGER_NAME)
    for handler in list(logger.handlers):
        logger.removeHandler(handler)
        handler.close()
    logger.propagate = False

    selected_mode = (mode or os.getenv("CASCAD_AI_LOG", DEFAULT_MODE)).strip().lower()
    if selected_mode not in VALID_MODES:
        valid = ", ".join(sorted(VALID_MODES))
        raise ValueError(
            f"Invalid logging mode '{selected_mode}'. Choose one of: {valid}"
        )

    if selected_mode == "off":
        logger.addHandler(logging.NullHandler())
        _configured = True
        return

    level_name = os.getenv("CASCAD_AI_LOG_LEVEL", DEFAULT_LEVEL).strip().upper()
    level = getattr(logging, level_name, logging.INFO)
    logger.setLevel(level)

    formatter = logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    if selected_mode in ("file", "both"):
        log_dir = _resolve_log_dir()
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        log_path = log_dir / f"cascad-ai-{timestamp}.log"
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    if selected_mode in ("console", "both"):
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    _configured = True
