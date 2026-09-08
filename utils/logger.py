"""
VisionAssist Logger
===================
Configures structured console logging with UTF-8 encoding support.
Prevents UnicodeEncodeError on Windows terminals with multilingual speech.
"""

import logging
import sys


def setup_logger(
    name: str = "VisionAssist",
    level: int = logging.INFO
) -> logging.Logger:
    """Sets up a thread-safe, UTF-8 capable console logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] [%(name)s] %(message)s",
            datefmt="%H:%M:%S"
        )
        stream = sys.stdout
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
        handler = logging.StreamHandler(stream)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger
