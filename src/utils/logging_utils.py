# src/utils/logging_utils.py
import logging
import os
from typing import Optional


def setup_logger(
    name: str = "mdm_project",
    log_level: int = logging.INFO,
    log_dir: Optional[str] = None,
) -> logging.Logger:
    """
    Create and configure a logger.

    Parameters
    ----------
    name : str
        Logger name.
    log_level : int
        Logging level, e.g. logging.INFO, logging.DEBUG.
    log_dir : Optional[str]
        If provided, logs will also be written to a file in this directory.

    Returns
    -------
    logging.Logger
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Avoid adding multiple handlers if setup_logger is called repeatedly
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # Optional file handler
    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"{name}.log")
        file_handler = logging.FileHandler(log_path, mode="a")
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Do not propagate to root logger (to avoid duplicate logs)
    logger.propagate = False

    return logger
