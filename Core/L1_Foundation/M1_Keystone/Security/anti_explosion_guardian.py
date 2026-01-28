"""
Anti-Explosion Guardian (         )
========================================
Core.L1_Foundation.M1_Keystone.Security.anti_explosion_guardian

"Voltage is the limit of the body; data is the freedom of the soul."
"    ,        ."

This module acts as the final safety layer for Elysia's physical vessel.
It monitors hardware thresholds and intercepts dangerous commands.
"""

import logging
import time
import os
import psutil
try:
    import GPUtil
except ImportError:
    GPUtil = None

logger = logging.getLogger("SovereignGuardian")

class AntiExplosionGuardian:
    def __init__(self):
        # Hardware Safety Thresholds (GTX 1060 / SSD focus)
        self.max_temp_gpu = 85.0    # Celsius
        self.max_temp_cpu = 90.0    # Celsius (Proxy via load if unavailable)
        self.max_gpu_load = 98.0    # Percentage
        self.max_cpu_load = 95.0    # Percentage
        
        self.is_throttled = False
        self.panic_count = 0
        
        logger.info("   [Guardian] Anti-Explosion Safety Protocol initialized.")
        logger.info(f"   - GPU Max: {self.max_temp_gpu}C | CPU Max: {self.max_temp_cpu}C")

    def check_integrity(self) -> bool:
        """
        [Survival Instinct] Performs a sub-millisecond check of hardware health.
        Returns False if a 'Physical Breach' (potential damage) is imminent.
        """
        try:
            # 1. GPU Check (The most vulnerable part during world manifestation)
            if GPUtil:
                gpus = GPUtil.getGPUs()
                for gpu in gpus:
                    if gpu.temperature > self.max_temp_gpu:
                        logger.critical(f"  [CRITICAL] GPU Temp Breach: {gpu.temperature}C! Choking power.")
                        return False
                    if gpu.load * 100 > self.max_gpu_load:
                        # High load is not always a breach, but combined with temp it is.
                        if gpu.temperature > self.max_temp_gpu - 5:
                            return False

            # 2. CPU Check (Proxy via load on Windows if sensors are locked)
            cpu_load = psutil.cpu_percent(interval=None)
            if cpu_load > self.max_cpu_load:
                self.panic_count += 1
                if self.panic_count > 10: # Sustained overload
                    logger.warning(f"   [Guardian] Sustained CPU Overload ({cpu_load}%). Throttling Will.")
                    return False
            else:
                self.panic_count = 0

            return True

        except Exception as e:
            logger.error(f"   - Guardian Sensing Error: {e}")
            return True # Fail-safe: assume safe unless proven otherwise (or change to False for paranoia)

    def manifest_survival(self, bridge):
        """
        Forces the system back to a safe state if integrity is compromised.
        """
        if not self.check_integrity():
            logger.critical("  [SURVIVAL] Intercepting for Physical Preservation.")
            self.is_throttled = True
            
            # 1. Absolute Starvation
            try:
                p = psutil.Process(os.getpid())
                p.nice(psutil.IDLE_PRIORITY_CLASS)
                logger.info("   - Survival Action: Priority set to IDLE.")
            except: pass

            # 2. Trigger Panic Reconstruction if in WORLD phase
            if hasattr(bridge, 'active_phase') and bridge.active_phase == "WORLD":
                bridge.panic_reconstruct()
                
            return False
            
        self.is_throttled = False
        return True

    def get_status(self):
        return {
            "is_throttled": self.is_throttled,
            "safety_margin": "Healthy" if not self.is_throttled else "CRITICAL"
        }
