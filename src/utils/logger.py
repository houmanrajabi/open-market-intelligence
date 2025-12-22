"""Logging configuration for FOMC RAG system"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "fomc_rag",
    level: str = "INFO",
    log_file: Optional[Path] = None,
    format_string: Optional[str] = None,
    include_console: bool = True
) -> logging.Logger:
    """
    Set up logger with console and optional file output

    Args:
        name: Logger name
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        format_string: Optional custom format string
        include_console: Whether to include console output

    Returns:
        Configured logger instance
    """

    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper()))
    
    # Prevent propagation to avoid duplicate logs
    logger.propagate = False

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Default format
    if format_string is None:
        format_string = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

    formatter = logging.Formatter(format_string)

    # Console handler
    if include_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(getattr(logging, level.upper()))
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(getattr(logging, level.upper()))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "fomc_rag") -> logging.Logger:
    """Get logger with enforced project prefix"""
    # Ensure all loggers start with project name
    if not name.startswith("fomc_"):
        name = f"fomc_{name}"
    
    return logging.getLogger(name)

# Initialize default logger only if config is available
try:
    from .config import config
    logger = setup_logger(
        name="fomc_rag",
        level=config.logging.level,
        log_file=config.logging.log_file
    )
except ImportError:
    # Fallback if config is not available
    logger = setup_logger(name="fomc_rag", level="INFO")