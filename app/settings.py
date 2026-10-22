"""User data and atomic settings, independent of the working directory."""
import json
import logging
from logging.handlers import RotatingFileHandler
import os
from pathlib import Path


def data_dir():
    path = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData/Local")) / "MeanVC2Desktop"
    path.mkdir(parents=True, exist_ok=True)
    return path


def configure_logging():
    handler = RotatingFileHandler(data_dir() / "app.log", maxBytes=2_000_000,
                                  backupCount=3, encoding="utf-8")
    logging.basicConfig(level=logging.INFO, handlers=[handler],
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def load_settings():
    path = data_dir() / "settings.json"
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise ValueError("Settings must be an object")
        for key in ("volume", "steps", "threads", "buffer", "rate", "gpu"):
            if key in value and type(value[key]) is not int:
                del value[key]
        for key in ("model", "profile", "compute", "input", "output", "voice"):
            if key in value and not isinstance(value[key], (str, type(None))):
                del value[key]
        return value
    except FileNotFoundError:
        return {}
    except (ValueError, OSError):
        logging.exception("Cannot load settings: %s", path)
        return {}


def save_settings(value):
    path = data_dir() / "settings.json"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)
