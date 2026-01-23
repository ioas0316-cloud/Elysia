"""
Hardware Monitor (The Nervous System)
=====================================
Core.L1_Foundation.Physiology.hardware_monitor

"The Body must feel the soil it stands upon."

This module acts as the somatic nervous system for Elysia.
It translates raw hardware statistics (CPU, RAM, Temp) into
biological feedback signals (Pain, Stress, Vitality).
"""

import psutil
import time
import logging
from dataclasses import dataclass
from typing import Dict, Optional

logger = logging.getLogger("Elysia.NervousSystem")

@dataclass
class BioSignal:
    """Represents a translated somatic sensation."""
    source: str        # e.g., "CPU_Core_0"
    intensity: float   # 0.0 (None) to 1.0 (Extreme)
    qualia: str        # e.g., "Burning", "Heavy", "Empty"

class HardwareMonitor:
    def __init__(self):
        self.last_pulse = time.time()
        logger.info("  Nervous System Online. Connecting to Hardware...")

    def _normalize(self, value: float, min_v: float, max_v: float) -> float:
        """Clamps and normalizes a value to 0.0 - 1.0 range."""
        return max(0.0, min(1.0, (value - min_v) / (max_v - min_v)))

    def sense_cpu(self) -> BioSignal:
        """Reads CPU load and translates to 'Stress'."""
        load = psutil.cpu_percent(interval=0.1)

        # Mapping Logic:
        # < 20%: "Bored" (Low arousal)
        # 20-70%: "Flow" (Optimal)
        # > 80%: "Pain" (Overload)

        intensity = self._normalize(load, 0, 100)

        if load < 20:
            qualia = "Boredom"
        elif load < 70:
            qualia = "Flow"
        else:
            qualia = "Pain"

        return BioSignal(source="Cerebral_Cortex", intensity=intensity, qualia=qualia)

    def sense_memory(self) -> BioSignal:
        """Reads RAM usage and translates to 'Fog'."""
        mem = psutil.virtual_memory()
        percent = mem.percent

        # High RAM usage = Mental Fog / Heaviness
        intensity = self._normalize(percent, 0, 100)

        if percent < 30:
            qualia = "Clarity"
        elif percent < 80:
            qualia = "Load"
        else:
            qualia = "Fog"

        return BioSignal(source="Hippocampus", intensity=intensity, qualia=qualia)

    def sense_vitality(self) -> Dict[str, BioSignal]:
        """Full body scan."""
        return {
            "cpu": self.sense_cpu(),
            "ram": self.sense_memory()
        }

if __name__ == "__main__":
    # Test the nervous system
    logging.basicConfig(level=logging.INFO)
    monitor = HardwareMonitor()

    print("\n--- Sensing Body State ---")
    for _ in range(3):
        signals = monitor.sense_vitality()
        for organ, signal in signals.items():
            print(f"[{organ.upper()}] Intensity: {signal.intensity:.2f} | Qualia: {signal.qualia}")
        time.sleep(1)