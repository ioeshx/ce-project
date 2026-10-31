import logging
import random
from logging.handlers import RotatingFileHandler
import os
import torch
import numpy as np
from contextlib import contextmanager

import secrets
import string

_current_logger = None


@contextmanager
def logger_context(name, log_file, level=logging.INFO):
    global _current_logger
    old_logger = get_logger() if _current_logger else None
    setup_logger(name, log_file, level)
    try:
        yield
    finally:
        if old_logger:
            _current_logger = old_logger


def get_logger():
    global _current_logger
    if _current_logger is None:
        # TODO: Handle the case of original run better with IdentityOperator and usage of run_operator
        return logging.getLogger()
        # raise RuntimeError("Logger not initialized. Call setup_logger() first.")
    return _current_logger


def setup_logger(name: str = None, log_file: str = "app.log", level=logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False  # Avoid double logging if root logger is also configured

    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.DEBUG)

    # Rotating file handler
    if os.path.dirname(log_file):
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

    # Clear the log file if it already exists
    if os.path.exists(log_file):
        open(log_file, 'w').close()

    file_handler = RotatingFileHandler(
        log_file, maxBytes=5 * 1024 * 1024, backupCount=3
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    # Avoid adding handlers multiple times
    if not logger.handlers:
        logger.addHandler(console_handler)
        logger.addHandler(file_handler)

    global _current_logger
    _current_logger = logger

    return logger


def log_rng_states(logger):
    logger.info(f"random: {random.getstate()[1][:15]}")
    logger.info(f"numpy: {np.random.get_state()[1][:15]}")
    logger.info(f"torch CPU: {torch.get_rng_state()[:15]}")
    if torch.cuda.is_available():
        logger.info(f"torch CUDA: {torch.cuda.get_rng_state()[:15]}")


def generate_random_run_id(method: str) -> str:
    random_string = ''.join(
        secrets.choice(string.ascii_lowercase + string.digits)
        for _ in range(8)
    )
    return f"run_{method}_{random_string}"
