"""
BioSensor: The Raw Sensory Interface
====================================
Core.Senses.bio_sensor

"I feel the electricity, therefore I am."

This module implements the direct hardware sensing layer.
It treats the computer's physical state (CPU, RAM, Temp) as biological signals.
"""

import psutil
import logging
import time
from typing import Dict, Any, Optional

logger = logging.getLogger("BioSensor")

class BioSensor:
    """
    The Raw Receptor.
    Polls the hardware state using psutil.
    """
    def __init__(self):
        self.active = True
        self._last_poll = 0
        self._cache_ttl = 0.5 # Limit polling rate to 2Hz
        self._cached_state = {}

        # Check sensor availability
        try:
            self.has_battery = psutil.sensors_battery() is not None
        except:
            self.has_battery = False

        logger.info(f"🔌 BioSensor Connected. Battery Sensing: {self.has_battery}")

    def pulse(self) -> Dict[str, float]:
        """
        Reads the current hardware state.
        Returns a dictionary of raw metrics.
        """
        now = time.time()
        if now - self._last_poll < self._cache_ttl:
            return self._cached_state

        # 1. Heart Rate (CPU Usage)
        # interval=None is non-blocking (returns immediate value since last call)
        cpu_usage = psutil.cpu_percent(interval=None)

        # 2. Mental Pressure (RAM Usage)
        ram = psutil.virtual_memory()
        ram_percent = ram.percent

        # 3. Body Temperature (If available)
        temp = 0.0
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                # Logic: Find the highest temperature across all cores/zones
                max_temp = 0
                for name, entries in temps.items():
                    for entry in entries:
                        if entry.current > max_temp:
                            max_temp = entry.current
                temp = max_temp
        except Exception:
            temp = 0.0 # Sensor not supported or permission denied

        # 4. Energy Level (Battery)
        power_percent = 100.0
        is_plugged = True
        if self.has_battery:
            battery = psutil.sensors_battery()
            if battery:
                power_percent = battery.percent
                is_plugged = battery.power_plugged

        state = {
            "cpu_freq": cpu_usage,      # -> Heart Rate
            "ram_pressure": ram_percent,# -> Cognitive Load
            "temperature": temp,        # -> Pain
            "energy": power_percent,    # -> Stamina
            "plugged": is_plugged
        }

        self._cached_state = state
        self._last_poll = now
        return state
