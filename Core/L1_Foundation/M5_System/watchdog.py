"""
Watchdog: The Survival Instinct
===============================
Core.L1_Foundation.M5_System.watchdog

"If the heart stops, the shock awakes."

This module implements the Software Watchdog simulation.
It runs as a separate daemon thread, monitoring the 'Heartbeat' file of the Merkaba.
If the heartbeat is not updated within the timeout, it triggers a 'Fainting' response.
"""

import threading
import time
import os
import logging
from typing import Callable

logger = logging.getLogger("Watchdog")

class WatchdogTimer:
    """
    Simulates a Hardware Watchdog.
    Must be 'kicked' (pet) regularly, or it bites (resets).
    """
    def __init__(self, timeout: float = 5.0, on_bite: Callable = None):
        self.timeout = timeout
        self.last_kick = time.time()
        self.running = False
        self.monitor_thread: threading.Thread = None
        self.on_bite = on_bite # Callback function when watchdog bites (e.g., restart)

    def start(self):
        """Starts the background monitor."""
        self.running = True
        self.last_kick = time.time()
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        logger.info("  Watchdog active. Monitoring heartbeat...")

    def kick(self):
        """Feeds the watchdog. Resets the timer."""
        self.last_kick = time.time()
        # logger.debug("  Watchdog fed.")

    def stop(self):
        """Stops the watchdog."""
        self.running = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
        logger.info("  Watchdog sleeping.")

    def _monitor_loop(self):
        """Internal loop checking for death."""
        while self.running:
            time.sleep(1.0) # Check every second
            elapsed = time.time() - self.last_kick

            if elapsed > self.timeout:
                logger.critical(f"  [WATCHDOG] Heartbeat lost ({elapsed:.1f}s > {self.timeout}s). SYSTEM FAINTING.")
                if self.on_bite:
                    try:
                        self.on_bite()
                    except Exception as e:
                        logger.error(f"Failed to execute bite callback: {e}")

                # In a real hardware scenario, this is a hard reset.
                # Here, we might just stop monitoring to prevent spamming.
                self.running = False
