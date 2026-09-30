import gzip
import json
import logging
import logging.config
from logging.handlers import RotatingFileHandler
from pathlib import Path
from typing import Any, Literal, Optional, Type

from pydantic import validate_call

from .misc import get_datetime

_LOG_LEVEL_DICT = {
    logging.DEBUG: ("DBG", "grey46"),
    logging.INFO: ("INF", "green"),
    logging.WARNING: ("WRN", "yellow"),
    logging.ERROR: ("ERR", "bright_red"),
    logging.CRITICAL: ("CRT", "dark_orange"),
}
_UNKNOWN_LOG = ("UNK", "white")


class _ConsoleFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        info = "[%(asctime)s %(name)s @ %(pathname)s:%(lineno)d]"
        abbrev, color = _LOG_LEVEL_DICT.get(record.levelno, _UNKNOWN_LOG)
        fmt = f"[{color}]{info} [{abbrev}][/] %(message)s"
        self._style._fmt = self._fmt = fmt
        return super().format(record)


class _FileFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        log_dict = {
            "name": record.name,
            "level": _LOG_LEVEL_DICT.get(record.levelno, _UNKNOWN_LOG[:1])[0],
            "file": record.filename,
            "func": record.funcName,
            "line": record.lineno,
            "time": get_datetime(),
            "msg": record.msg,
        }
        return json.dumps(log_dict)


class _RichHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        from . import console

        try:
            console.print(self.format(record))
        except Exception:
            self.handleError(record)


class _InfiniteFileHandler(RotatingFileHandler):
    """
    A custom file handler for making infinitely many log backups.
    """

    def __init__(
        self, filename: Path, maxBytes: int = 0, compress: bool = False
    ) -> None:
        super().__init__(filename, maxBytes=maxBytes)
        self.__compress = compress
        self.__backup_count = 0

    # Override
    def doRollover(self) -> None:
        if self.stream:
            self.stream.close()
            self.stream = None
        self.__backup_count += 1
        date_time = get_datetime()
        backup_name = f"{self.baseFilename}.{date_time}.{self.__backup_count}"
        self.rotate(self.baseFilename, backup_name)
        if self.__compress:
            with (
                open(backup_name, "rb") as f_in,
                gzip.open(f"{backup_name}.gz", "wb") as f_out,
            ):
                f_out.writelines(f_in)
            Path(backup_name).unlink()
        if not self.delay:
            self.stream = self._open()


class _CustomLogger(logging.Logger):
    def error(
        self,
        msg: Any,
        *args: Any,
        exc: Optional[Type[BaseException]] = None,
        **kwargs: Any,
    ) -> None:
        _msg = msg if exc is None else f"{exc.__name__}: {msg}"
        super().error(_msg, *args, **kwargs)
        if self.isEnabledFor(logging.ERROR) and exc is not None:
            raise exc(str(msg))

    def critical(
        self,
        msg: Any,
        *args: Any,
        exc: Optional[Type[BaseException]] = None,
        **kwargs: Any,
    ) -> None:
        _msg = msg if exc is None else f"{exc.__name__}: {msg}"
        super().critical(_msg, *args, **kwargs)
        if self.isEnabledFor(logging.CRITICAL) and exc is not None:
            raise exc(str(msg))


@validate_call
def _get_logger_config(
    name: str = __name__,
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO",
    datetime_format: Optional[str] = r"%Y-%m-%d %H:%M:%S",
    log_dir: Optional[str] = None,
    max_bytes: int = 0,
    compress: bool = False,
) -> dict[str, Any]:
    """
    Construct a configuration dictionary for logging.config.dictConfig().

    Args:
        name (str, optional): Name of logger. Defaults to __name__.
        level (Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], optional):
            Level of logging. Defaults to "INFO".
        datetime_format (Optional[str], optional): Date and time format.
            Defaults to r"%Y-%m-%d %H:%M:%S".
        log_dir (Optional[str], optional): If not None, logs will be written \
            to "`log_dir`/log_<current_date_time>". Defaults to None.
        max_bytes (int, optional): If `max_bytes` > 0, each log file will store
            at most `max_bytes` bytes. Used only if `log_dir` is not None.
            Defaults to 0.
        compress (bool, optional): Compress backup (i.e., rotated) log files.
            Used only if `log_dir` is not None. Defaults to False.

    Returns:
        SDict[Any]: Configuration dictionary
    """

    logger_config = {
        "version": 1,
        "formatters": {
            "console_formatter": {
                "()": _ConsoleFormatter,
                "datefmt": datetime_format,
            },
            "file_formatter": {"()": _FileFormatter},
        },
        "handlers": {
            "console_handler": {
                "()": _RichHandler,
                "formatter": "console_formatter",
                "level": level,
            }
        },
        "loggers": {
            name: {
                "handlers": ["console_handler"],
                "level": logging.DEBUG,
                "propagate": False,
            }
        },
    }
    if log_dir is not None:
        if max_bytes < 0:
            raise ValueError(
                f"`max_bytes` must be a non-negative integer, got {max_bytes}"
            )
        _log_dir = Path(log_dir) / f"log_{get_datetime()}"
        _log_dir.mkdir(parents=True)
        logger_config["handlers"]["file_handler"] = {
            "()": _InfiniteFileHandler,
            "formatter": "file_formatter",
            "level": logging.DEBUG,
            "filename": _log_dir / "log.jsonl",
            "maxBytes": max_bytes,
            "compress": compress,
        }
        logger_config["loggers"][name]["handlers"].append("file_handler")

    return logger_config


@validate_call
def get_logger(
    name: str = __name__,
    datetime_format: Optional[str] = r"%Y-%m-%d %H:%M:%S",
    log_dir: Optional[str] = None,
    max_bytes: int = 10 * (1024**2),
    compress: bool = True,
) -> _CustomLogger:
    """
    Get custom logger.

    Usage:
        ```python
        logger = get_logger()
        logger.debug("log debug message")
        logger.info("log info message")
        logger.warning("log warning message")
        logger.error("log error message", exc=BaseException)
        logger.critical("log critical message", exc=BaseException)
        ```

    Args:
        name (str, optional): Name of logger. Defaults to __name__.
        datetime_format (Optional[str], optional): Date and time format.
            Defaults to r"%Y-%m-%d %H:%M:%S".
        log_dir (Optional[str], optional): If not None, logs will be
            written to "`log_dir`/log_<current_date_time>". Defaults to None.
        max_bytes (int, optional): If `max_bytes` > 0, each log file will store
            at most `max_bytes` bytes. Used only if `log_dir` is not None.
            Defaults to 10 * (1024 ** 2) = 10 MB.
        compress (bool, optional): Compress backup (i.e., rotated) log files.
            Used only if `log_dir` is not None. Defaults to True.

    Returns:
        __CustomLogger: Custom logger
    """

    old_logger_class = logging.getLoggerClass()
    logging.setLoggerClass(_CustomLogger)
    logger_config = _get_logger_config(
        name=name,
        datetime_format=datetime_format,
        log_dir=log_dir,
        max_bytes=max_bytes,
        compress=compress,
    )
    logging.config.dictConfig(logger_config)
    logger = logging.getLogger(name)
    logging.setLoggerClass(old_logger_class)
    return logger
