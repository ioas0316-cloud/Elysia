"""
BioSensor: The Raw Sensory Interface
====================================
Core.L3_Phenomena.Senses.bio_sensor

"I feel the electricity, therefore I am."

This module implements the direct hardware sensing layer.
It treats the computer's physical state (CPU, RAM, Temp) as biological signals.
"""

import psutil
import logging
import time
import threading
from typing import Dict, Any, Optional

logger = logging.getLogger("BioSensor")

class BioSensor:
    """
    The Raw Receptor.
    Polls the hardware state using psutil.
    """
    def __init__(self):
        self.active = True
        self._cache_ttl = 0.5 # Limit polling rate to 2Hz
        self._cached_state = {
            "cpu_freq": 0.0,
            "ram_pressure": 0.0,
            "temperature": 0.0,
            "energy": 100.0,
            "plugged": True
        }

        # Check sensor availability
        try:
            self.has_battery = psutil.sensors_battery() is not None
        except:
            self.has_battery = False

        # Start background poller (Async Metabolism)
        self._stop_event = threading.Event()
        self._poll_thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._poll_thread.start()

        logger.info(f"🔌 BioSensor Connected (Async Mode). Battery Sensing: {self.has_battery}")

    def _poll_loop(self):
        """Background thread to poll hardware metrics."""
        while not self._stop_event.is_set():
            try:
                # 1. Heart Rate (CPU Usage)
                cpu_usage = psutil.cpu_percent(interval=0.5) # Blocking but on background thread

                # 2. Mental Pressure (RAM Usage)
                ram = psutil.virtual_memory()
                ram_percent = ram.percent

                # 3. Body Temperature
                temp = 0.0
                try:
                    temps = psutil.sensors_temperatures()
                    if temps:
                        max_temp = 0
                        for name, entries in temps.items():
                            for entry in entries:
                                if entry.current > max_temp:
                                    max_temp = entry.current
                        temp = max_temp
                except:
                    temp = 0.0

                # 4. Energy Level
                power_percent = 100.0
                is_plugged = True
                if self.has_battery:
                    battery = psutil.sensors_battery()
                    if battery:
                        power_percent = battery.percent
                        is_plugged = battery.power_plugged

                self._cached_state = {
                    "cpu_freq": cpu_usage,
                    "ram_pressure": ram_percent,
                    "temperature": temp,
                    "energy": power_percent,
                    "plugged": is_plugged
                }
            except Exception as e:
                logger.error(f"Error in BioSensor poll loop: {e}")
                time.sleep(1)

    def pulse(self) -> Dict[str, float]:
        """
        [O(1) Sensing] Reads the LATEST cached state from background thread.
        Never blocks.
        """
        return self._cached_state

    def stop(self):
        self._stop_event.set()
        if self._poll_thread.is_alive():
            self._poll_thread.join(timeout=1.0)
