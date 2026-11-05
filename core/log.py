"""Project-wide logging.
e.g.
    from core.log import get_logger

    logger = get_logger(__name__)
    logger.debug("per-frame detail")
    logger.trace("per-detection detail")

Levels
------

======  ===============================================================
TRACE   raw model output
DEBUG   one sum line per frame
INFO    basic info
WARNING problems (lost calibration, unreadable frame)
======  ===============================================================

INFO by default

Selecting what is printed
-------------------------
A per-module override (``ball_tracker=DEBUG``) beats
the global level, and modules outside this project are only shown when named
explicitly.

===========================  ===============================================
``FOOTBALL_LOG_LEVEL``        global level (``INFO`` by default)
``FOOTBALL_LOG_MODULES``      comma-separated ``module=LEVEL`` overrides
``FOOTBALL_LOG_FILE``         also write to this file (DEBUG by default)
``FOOTBALL_LOG_FILE_LEVEL``   file handler level (e.g. ``TRACE``)
``NO_COLOR``                  any value disables ANSI colours
===========================  ===============================================

Frame context
-------------
``set_frame_context(frame_idx)`` tags every record emitted afterwards with
``[f=0042]``, without every function needing to pass the frame index around.
``clear_frame_context()`` removes the tag.
"""

from __future__ import annotations

import contextvars
import logging
import os
import sys
from pathlib import Path
from typing import Mapping, Sequence

__all__ = [
    "TRACE",
    "configure_logging",
    "get_logger",
    "parse_module_levels",
    "set_frame_context",
    "clear_frame_context",
]

TRACE = 5
logging.addLevelName(TRACE, "TRACE")


def _trace(self, message, *args, **kwargs):
    if self.isEnabledFor(TRACE):
        self._log(TRACE, message, args, **kwargs)


# Module-level loggers get ``logger.trace(...)`` alongside the stdlib methods.
logging.Logger.trace = _trace  # type: ignore[attr-defined]

_LEVELS = {
    "TRACE": TRACE,
    "DEBUG": logging.DEBUG,
    "INFO": logging.INFO,
    "WARNING": logging.WARNING,
    "ERROR": logging.ERROR,
    "CRITICAL": logging.CRITICAL,
}

_LEVEL_ABBREV = {
    TRACE: "TRC",
    logging.DEBUG: "DBG",
    logging.INFO: "INF",
    logging.WARNING: "WRN",
    logging.ERROR: "ERR",
    logging.CRITICAL: "CRT",
}

_LEVEL_COLOURS = {
    TRACE: "\033[90m",        # bright black
    logging.DEBUG: "\033[36m",  # cyan
    logging.INFO: "\033[32m",   # green
    logging.WARNING: "\033[33m",  # yellow
    logging.ERROR: "\033[31m",    # red
    logging.CRITICAL: "\033[1;31m",
}
_RESET = "\033[0m"

#: Loggers created by this project. Everything else is third-party and only
#: appears when explicitly named in ``modules``.
_OUR_PREFIXES = ("core", "__main__")

#: Third-party loggers that are noisy even at INFO.
_NOISY_LOGGERS = ("ultralytics", "torch", "torchvision", "PIL", "matplotlib", "urllib3")

_ENV_LEVEL = "FOOTBALL_LOG_LEVEL"
_ENV_MODULES = "FOOTBALL_LOG_MODULES"
_ENV_FILE = "FOOTBALL_LOG_FILE"
_ENV_FILE_LEVEL = "FOOTBALL_LOG_FILE_LEVEL"

_frame_context: contextvars.ContextVar[int | None] = contextvars.ContextVar(
    "football_log_frame", default=None
)

_module_levels: dict[str, int] = {}


# --------------------------------------------------------------------------- API


def get_logger(name: str) -> logging.Logger:
    """Return the logger for ``name`` (normally ``__name__``).

    Any per-module override registered by :func:`configure_logging` is applied
    here too, so a logger created after configuration behaves the same as one
    that already existed.
    """
    logger = logging.getLogger(name)
    override = _override_for(name)
    if override is not None:
        logger.setLevel(override)
    return logger


def set_frame_context(frame_idx: int) -> None:
    """Tag every subsequent log record with the frame being processed."""
    _frame_context.set(int(frame_idx))


def clear_frame_context() -> None:
    """Stop tagging records with a frame index."""
    _frame_context.set(None)


def parse_module_levels(specs: "str | Sequence[str]") -> dict[str, int]:
    """Parse ``["ball_tracker=DEBUG", "calib=TRACE"]`` into a level mapping.

    A single comma- or space-separated string is accepted as well, which is the
    form used by ``FOOTBALL_LOG_MODULES``.
    """
    if isinstance(specs, str):
        specs = specs.replace(",", " ").split()

    levels: dict[str, int] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"module level must be NAME=LEVEL, got {spec!r}")
        name, value = spec.split("=", 1)
        name = name.strip()
        if not name:
            raise ValueError(f"module level must be NAME=LEVEL, got {spec!r}")
        levels[name] = _parse_level(value)
    return levels


def configure_logging(
    level: "str | int | None" = None,
    log_file: "str | os.PathLike | None" = None,
    modules: "Mapping[str, str | int] | str | Sequence[str] | None" = None,
    *,
    console: bool = True,
    file_level: "str | int | None" = None,
    color: bool | None = None,
    timestamp: bool = True,
) -> None:
    """Install the project's handlers on the root logger.

    Safe to call more than once: handlers installed by an earlier call are
    removed first, so re-configuring from a CLI or a notebook does not stack
    handlers.

    Args:
        level: global console level. Defaults to ``FOOTBALL_LOG_LEVEL``, then
            ``INFO``.
        log_file: also append to this file. Defaults to
            ``FOOTBALL_LOG_FILE``. Parent directories are created.
        modules: per-module levels, as a mapping or a sequence of
            ``NAME=LEVEL`` strings. Defaults to ``FOOTBALL_LOG_MODULES``.
            ``NAME`` may be the full logger name (``core.ball_tracker``) or
            its last component (``ball_tracker``).
        console: set to ``False`` for file-only logging.
        file_level: file handler level. Defaults to
            ``FOOTBALL_LOG_FILE_LEVEL``, then DEBUG so a log file captures
            per-frame detail without the console noise. Set to ``"TRACE"`` to
            also capture per-detection output.
        color: force ANSI colours on or off. By default they are on when
            stderr is a terminal and ``NO_COLOR`` is unset.
        timestamp: include the wall-clock time in console lines.
    """
    global _module_levels

    console_level = _parse_level(level if level is not None else os.environ.get(_ENV_LEVEL, "INFO"))
    log_file = log_file if log_file is not None else os.environ.get(_ENV_FILE)
    file_level = _parse_level(
        file_level if file_level is not None else os.environ.get(_ENV_FILE_LEVEL, "DEBUG")
    )

    resolved_modules: dict[str, int] = {}
    env_modules = os.environ.get(_ENV_MODULES)
    if env_modules:
        resolved_modules.update(parse_module_levels(env_modules))
    if modules is not None:
        if isinstance(modules, Mapping):
            resolved_modules.update({name: _parse_level(value) for name, value in modules.items()})
        else:
            resolved_modules.update(parse_module_levels(modules))
    _module_levels = resolved_modules

    root = logging.getLogger()
    _remove_project_handlers(root)

    # Records must reach the handlers before their filters decide. Keep the
    # root low enough for every project logger to pass its own level check.
    project_level = min(
        [console_level, file_level if log_file else console_level]
        + list(resolved_modules.values())
    )
    root.setLevel(project_level)

    if color is None:
        color = not os.environ.get("NO_COLOR") and sys.stderr.isatty()

    if console:
        handler = _TqdmStreamHandler(sys.stderr)
        handler.setLevel(logging.NOTSET)
        handler.setFormatter(_ConsoleFormatter(color=color, timestamp=timestamp))
        handler.addFilter(_ContextFilter())
        handler.addFilter(_RecordFilter(console_level))
        _mark_project_handler(handler)
        root.addHandler(handler)

    if log_file:
        path = Path(log_file).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        handler = logging.FileHandler(path, encoding="utf-8")
        handler.setLevel(logging.NOTSET)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s.%(msecs)03d %(levelname)-8s %(name)s%(frame_tag)s %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        handler.addFilter(_ContextFilter())
        handler.addFilter(_RecordFilter(file_level))
        _mark_project_handler(handler)
        root.addHandler(handler)

    # Give every project logger the lowest level any handler could emit, so
    # ``logger.debug(...)`` returns early when nothing is listening.
    matched_overrides: set[str] = set()
    for name, logger in list(logging.root.manager.loggerDict.items()):
        if not isinstance(logger, logging.Logger):
            continue
        if _is_project_logger(name):
            logger.setLevel(_level_for(name, project_level))
        for key, key_level in resolved_modules.items():
            if _matches(name, key):
                logger.setLevel(key_level)
                matched_overrides.add(key)

    # An override for a logger that does not exist yet (a lazily imported
    # third-party module) still has to be applied once it is created.
    for key, key_level in resolved_modules.items():
        if key not in matched_overrides:
            logging.getLogger(key).setLevel(key_level)

    # Third-party loggers stay quiet unless named explicitly; this also keeps
    # the file handler from collecting framework debug output.
    for name in _NOISY_LOGGERS:
        if _override_for(name) is None:
            logging.getLogger(name).setLevel(logging.WARNING)


# ---------------------------------------------------------------------- levels


def _parse_level(value: "str | int") -> int:
    if isinstance(value, int):
        return value
    key = str(value).strip().upper()
    if key in _LEVELS:
        return _LEVELS[key]
    raise ValueError(f"unknown log level {value!r}; expected one of {', '.join(_LEVELS)}")


def _override_for(name: str) -> int | None:
    for key, level in _module_levels.items():
        if _matches(name, key):
            return level
    return None


def _level_for(name: str, fallback: int) -> int:
    override = _override_for(name)
    return fallback if override is None else override


def _matches(name: str, key: str) -> bool:
    """Match a full logger name or its last component (``ball_tracker``)."""
    if name == "__main__" and key == "main":
        return True
    return name == key or name.endswith("." + key)


def _is_project_logger(name: str) -> bool:
    return any(name == prefix or name.startswith(prefix + ".") for prefix in _OUR_PREFIXES)


def _short_name(name: str) -> str:
    if name == "__main__":
        return "main"
    return name.rsplit(".", 1)[-1]


# ------------------------------------------------------------------- handlers


class _ContextFilter(logging.Filter):
    """Attach the current frame index to every record."""

    def filter(self, record: logging.LogRecord) -> bool:
        frame = _frame_context.get()
        record.frame = frame
        record.frame_tag = f" [f={frame:04d}]" if frame is not None else ""
        return True


class _RecordFilter(logging.Filter):
    """Scope + per-module level filtering, applied per handler.

    Handler-level thresholds alone cannot express "module A at DEBUG, module B
    at INFO" because every record would have to pass one shared level.
    """

    def __init__(self, default_level: int):
        super().__init__()
        self.default_level = default_level

    def filter(self, record: logging.LogRecord) -> bool:
        name = record.name
        named_override = any(_matches(name, key) for key in _module_levels)
        if not (_is_project_logger(name) or named_override):
            return False
        return record.levelno >= _level_for(name, self.default_level)


class _TqdmStreamHandler(logging.StreamHandler):
    """Write through ``tqdm.write`` so log lines do not tear the progress bar."""

    def emit(self, record: logging.LogRecord) -> None:
        try:
            message = self.format(record)
        except Exception:
            self.handleError(record)
            return
        try:
            from tqdm import tqdm

            tqdm.write(message, file=self.stream)
            return
        except Exception:
            pass
        try:
            stream = self.stream or sys.stderr
            stream.write(message + "\n")
            stream.flush()
        except Exception:
            self.handleError(record)


class _ConsoleFormatter(logging.Formatter):
    """``HH:MM:SS.mmm LVL module [f=0001] message``, optionally coloured."""

    def __init__(self, color: bool, timestamp: bool):
        super().__init__()
        self.color = color
        self.timestamp = timestamp

    def format(self, record: logging.LogRecord) -> str:
        level_name = _LEVEL_ABBREV.get(record.levelno, record.levelname[:3])
        if self.color:
            colour = _LEVEL_COLOURS.get(record.levelno, "")
            level_name = f"{colour}{level_name}{_RESET}"

        module = _short_name(record.name)
        prefix = f"{level_name} {module:<17}"
        if self.timestamp:
            timestamp = self.formatTime(record, "%H:%M:%S")
            prefix = f"{timestamp}.{int(record.msecs):03d} {prefix}"
        frame_tag = getattr(record, "frame_tag", "")
        message = record.getMessage()
        if record.exc_info:
            message = f"{message}\n{self.formatException(record.exc_info)}"
        return f"{prefix}{frame_tag} {message}"


# ------------------------------------------------------------------- internals


def _mark_project_handler(handler: logging.Handler) -> None:
    handler._football_log_handler = True  # type: ignore[attr-defined]


def _remove_project_handlers(root: logging.Logger) -> None:
    for handler in list(root.handlers):
        if getattr(handler, "_football_log_handler", False):
            root.removeHandler(handler)
            handler.close()
