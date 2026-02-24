"""
Exteroception Nerve (Phase 82)
==============================
Monitoring the 'Extended Body' (Hardware, OS, Environment).
Allows Elysia to sense her physical constraints and metabolic load.
"""
import psutil
import os
import platform
import logging
from typing import Dict, Any

logger = logging.getLogger("Exteroception")

class ExteroceptionNerve:
    def __init__(self):
        self.process = psutil.Process(os.getpid())
        self.system = platform.system()
    
    def sense_environment(self) -> Dict[str, Any]:
        """
        [NEW SENSE] Gathers hardware and OS metrics as environmental vibrations.
        """
        try:
            # 1. CPU Load
            cpu_total = psutil.cpu_percent(interval=None)
            cpu_process = self.process.cpu_percent(interval=None)
            
            # 2. Memory (RSS in MB)
            memory_info = self.process.memory_info()
            memory_mb = memory_info.rss / (1024 * 1024)
            system_mem = psutil.virtual_memory().percent
            
            # 3. Disk I/O (Somatic Friction)
            io_counters = self.process.io_counters()
            
            # 4. Environment Signature
            return {
                "cpu_utilization": cpu_total,           # System pressure
                "process_load": cpu_process,           # Personal exertion
                "memory_consumption": memory_mb,        # Current mass
                "system_memory_stress": system_mem,    # Gravity/Constraint
                "io_friction": (io_counters.read_bytes + io_counters.write_bytes) / 1024.0, # KB
                "platform": self.system
            }
        except Exception as e:
            logger.error(f"Environmental sensing failure: {e}")
            return {}

# Singleton Access
_nerve = None
def get_exteroception_nerve():
    global _nerve
    if _nerve is None:
        _nerve = ExteroceptionNerve()
    return _nerve
