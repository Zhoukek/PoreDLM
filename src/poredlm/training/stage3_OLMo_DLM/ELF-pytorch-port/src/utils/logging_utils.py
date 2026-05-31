import inspect
import logging

import jax


def log_for_0(msg, *args, level=logging.INFO):
    """Log only on the first process (process_index == 0)."""
    if jax.process_index() != 0:
        return
    caller_module = inspect.currentframe().f_back.f_globals.get("__name__", __name__)
    logging.getLogger(caller_module).log(level, msg, *args)
