#!/usr/bin/env python
# coding: utf-8
"""
Logging utilities for cs_copilot.

This module provides logging configuration and utilities to handle
common logging issues in Jupyter notebooks and multiprocessing environments.
"""

import logging
import warnings
from collections.abc import Mapping, Sequence
from typing import Optional


def setup_logging(
    level: int = logging.INFO,
    suppress_warnings: bool = True,
    suppress_tqdm: bool = True,
    force: bool = False,
) -> logging.Logger:
    """
    Set up logging configuration for cs_copilot.

    Args:
        level: Logging level (default: INFO)
        suppress_warnings: Whether to suppress common warnings
        suppress_tqdm: Whether to suppress tqdm-related warnings
        force: Whether to reconfigure logging even if handlers already exist

    Returns:
        Configured logger instance
    """
    # Configure logging
    root_logger = logging.getLogger()
    if force or not root_logger.handlers:
        logging.basicConfig(
            level=level,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            force=force,
        )
    else:
        root_logger.setLevel(level)

    logger = logging.getLogger("cs_copilot")

    if suppress_warnings:
        # Suppress CUDA warnings
        warnings.filterwarnings("ignore", message="Can't initialize NVML")
        warnings.filterwarnings("ignore", category=UserWarning, module="torch.cuda")

        # Suppress other common warnings
        warnings.filterwarnings("ignore", message=".*torch.*", category=FutureWarning)
        warnings.filterwarnings("ignore", message=".*numpy.*", category=DeprecationWarning)

    if suppress_tqdm:
        # Suppress tqdm warnings that occur in Jupyter notebooks
        warnings.filterwarnings("ignore", message=".*tqdm.*")

        # Try to fix tqdm issues by setting environment variable
        import os

        os.environ["TQDM_DISABLE"] = "1"

    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """
    Get a logger instance for the given name.

    Args:
        name: Logger name (default: 'cs_copilot')

    Returns:
        Logger instance
    """
    if name is None:
        name = "cs_copilot"
    return logging.getLogger(name)


def compact_log_data(
    value,
    *,
    max_items: int = 6,
    max_string_length: int = 120,
    depth: int = 0,
    max_depth: int = 2,
) -> str:
    """
    Build a compact, single-line representation of runtime data for logs.

    The goal is observability, not perfect serialization: keep the most useful
    fields visible while avoiding giant payloads in terminal logs.
    """
    if value is None:
        return "-"

    if isinstance(value, str):
        sanitized = " ".join(value.split())
        if len(sanitized) > max_string_length:
            sanitized = f"{sanitized[: max_string_length - 3]}..."
        return sanitized

    if isinstance(value, (int, float, bool)):
        return str(value)

    if depth >= max_depth:
        return f"<{type(value).__name__}>"

    if isinstance(value, Mapping):
        parts = []
        items = list(value.items())
        for key, item in items[:max_items]:
            parts.append(
                f"{key}={compact_log_data(item, max_items=max_items, max_string_length=max_string_length, depth=depth + 1, max_depth=max_depth)}"
            )
        if len(items) > max_items:
            parts.append(f"...+{len(items) - max_items} keys")
        return "{" + ", ".join(parts) + "}"

    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        items = list(value)
        preview = [
            compact_log_data(
                item,
                max_items=max_items,
                max_string_length=max_string_length,
                depth=depth + 1,
                max_depth=max_depth,
            )
            for item in items[:max_items]
        ]
        if len(items) > max_items:
            preview.append(f"...+{len(items) - max_items} items")
        return "[" + ", ".join(preview) + "]"

    return compact_log_data(str(value), max_items=max_items, max_string_length=max_string_length)
