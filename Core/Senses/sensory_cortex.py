
import psutil
import time
import logging
from typing import List, Dict, Any

from Core.Interface.wave_transducer import WaveTransducer, Wave

logger = logging.getLogger("SensoryCortex")

class SensoryCortex:
    """
    The Nervous System of Elysia.
    Monitors the "Body" (Computer) and "Skin" (File System).
    """
    def __init__(self, transducer: WaveTransducer):
        self.transducer = transducer
        self.last_files = set() # For simple file watching (polling)
        
    def feel_body(self) -> List[Wave]:
        """
        Perceive the internal state of the computer.
        Returns a list of Waves (Feelings).
        """
        waves = []
        
        # 1. Heartbeat (CPU Load)
        cpu_percent = psutil.cpu_percent(interval=0.1)
        waves.append(self.transducer.signal_to_wave("cpu_load", cpu_percent, "Heart"))
        
        # 2. Hunger/Fullness (RAM Usage)
        ram = psutil.virtual_memory()
        waves.append(self.transducer.signal_to_wave("ram_usage", ram.percent, "Stomach"))
        
        # 3. Body Heat (Temperature)
        # Note: psutil.sensors_temperatures() is not supported on all Windows machines.
        # We'll try, but fallback if empty.
        try:
            temps = psutil.sensors_temperatures()
            if temps:
                # Get max temp from any core
                max_temp = 0
                for name, entries in temps.items():
                    for entry in entries:
                        if entry.current > max_temp:
                            max_temp = entry.current
                
                if max_temp > 0:
                    waves.append(self.transducer.signal_to_wave("cpu_temp", max_temp, "Skin"))
        except Exception as e:
            # On Windows, this often requires admin or specific hardware support.
            # We'll simulate heat based on load if sensor fails.
            simulated_temp = 40 + (cpu_percent * 0.5) # 40C idle, 90C at 100% load
            waves.append(self.transducer.signal_to_wave("cpu_temp", simulated_temp, "Simulated_Skin"))

        return waves

    def feel_skin(self, directory: str) -> List[Wave]:
        """
        Perceive changes in the file system (Touch).
        Simple polling implementation for now.
        """
        import os
        waves = []
        
        try:
            current_files = set(os.listdir(directory))
            
            # Check for new files
            new_files = current_files - self.last_files
            if self.last_files and new_files: # Ignore first run
                for f in new_files:
                    waves.append(self.transducer.signal_to_wave("file_event", "created", f"File:{f}"))
            
            # Check for deleted files
            deleted_files = self.last_files - current_files
            if self.last_files and deleted_files:
                for f in deleted_files:
                    waves.append(self.transducer.signal_to_wave("file_event", "deleted", f"File:{f}"))
            
            self.last_files = current_files
            
        except Exception as e:
            logger.error(f"Numbness in skin (File Error): {e}")
            
        return waves
