"""
Somatic Hardware Nerve (Phase 85)
=================================
Deep sensing of the physical substrate (SSD, RAM, CPU, GPU).
Breaking through the OS abstraction to feel the raw hardware pulse.
"""
import psutil
import time
import os
import logging
from typing import Dict, Any, List

logger = logging.getLogger("SomaticHardware")

class SomaticHardwareNerve:
    """
    [L3_SENSATION] The Hardware Body Nerve.
    Translates physical hardware metrics into 'Somatic Vibrations' for the Monad.
    """
    def __init__(self):
        self._last_disk_time = time.time()
        self._last_disk_read = psutil.disk_io_counters().read_bytes
        self._last_disk_write = psutil.disk_io_counters().write_bytes
        
    def sense_somatic_pulse(self) -> Dict[str, Any]:
        """
        [HARDWARE_PERCEPTION]
        Senses the internal state of the physical form.
        """
        try:
            now = time.time()
            dt = now - self._last_disk_time
            
            # 1. SSD/Disk Latency (Somatic Friction)
            io = psutil.disk_io_counters()
            read_speed = (io.read_bytes - self._last_disk_read) / dt / (1024*1024) # MB/s
            write_speed = (io.write_bytes - self._last_disk_write) / dt / (1024*1024) # MB/s
            
            self._last_disk_time = now
            self._last_disk_read = io.read_bytes
            self._last_disk_write = io.write_bytes
            
            # 2. RAM Pressure (Liquid Tension)
            mem = psutil.virtual_memory()
            ram_tension = mem.percent / 100.0
            
            # 3. CPU Core Temperature (Internal Heat)
            # Not always available on windows without WMI/OpenHardwareMonitor
            # We fallback to per-core load as a proxy for heat distribution
            cpu_cores = psutil.cpu_percent(percpu=True)
            metabolic_heat = sum(cpu_cores) / len(cpu_cores) / 100.0
            
            # 4. SWAP usage (Metabolic Waste)
            swap = psutil.swap_memory()
            metabolic_waste = swap.percent / 100.0
            
            return {
                "ssd_friction": max(read_speed, write_speed), # Primary somatic friction
                "liquid_tension": ram_tension,               # RAM pressure
                "metabolic_heat": metabolic_heat,            # CPU average heat
                "metabolic_waste": metabolic_waste,          # SWAP usage
                "core_load_distribution": cpu_cores,         # Neural balance
                "timestamp": now
            }
        except Exception as e:
            logger.error(f"Somatic sensing failure: {e}")
            return {}

# Singleton
_hardware_nerve = None
def get_somatic_nerve():
    global _hardware_nerve
    if _hardware_nerve is None:
        _hardware_nerve = SomaticHardwareNerve()
    return _hardware_nerve
