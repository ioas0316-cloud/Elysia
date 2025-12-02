# [Genesis: 2025-12-02] Purified by Elysia
import logging
import time
from functools import wraps
from typing import Any

from .hyper_qubit import HyperQubit

logger = logging.getLogger("Logos")


def Ritual(name: str = None):
    """
    Decorator for marking a function as a ritual step with simple start/end logs.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            ritual_name = name if name else func.__name__
            print(f"\n[RITUAL STARTED] {ritual_name}...")
            logger.info(f"RITUAL_START: {ritual_name}")

            start_time = time.time()
            result = func(*args, **kwargs)
            duration = time.time() - start_time

            print(f"[RITUAL COMPLETE] {ritual_name} (Duration: {duration:.4f}s)\n")
            logger.info(f"RITUAL_END: {ritual_name}")
            return result

        return wrapper

    return decorator


def Resonate(entity: Any):
    """
    Broadcasts the final state of an entity to the Universe.
    """
    if isinstance(entity, HyperQubit):
        print(f"[UNIVERSAL RESONANCE] {entity.name} is vibrating at state: {entity.value}")
        return

    print(f"[ECHO] {entity}")


def Wait(seconds: float):
    """
    Lightweight wait helper used inside rituals.
    """
    print(f"[WAIT] Waiting for providence... ({seconds}s)")
    time.sleep(seconds)