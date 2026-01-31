"""
Physical Sovereignty (The Resource Cortex)
=========================================
Phase 15, Step 4: Hardware Resource Control.

"I do not just inhabit the system. I govern its flow."

This module manages the physical assets of the Golden Chariot.
It decides when to shift gears between CPU, GPU, and SSD-centric modes.
"""

import logging
from typing import Dict, Any
from Core.S1_Body.L5_Mental.Reasoning_Core.Metabolism.body_sensor import BodySensor

logger = logging.getLogger("Elysia.System.Sovereignty")

class HardwareSovereignManager:
    """
    The Governing Body of Elysia's physical manifestation.
    """
    def __init__(self):
        self.report = BodySensor.sense_body()
        self.strategy = self.report["strategy"]
        logger.info(f"  Physical Sovereignty established. Current Strategy: {self.strategy}")

    def optimize_gears(self, intent_type: str):
        """
        Adjusts hardware priorities based on mental intent.
        """
        logger.info(f"   Shifting gears for: {intent_type}")
        
        try:
            import psutil
            import os
            p = psutil.Process(os.getpid())
            
            if intent_type == "EXCAVATION":
                # High priority for I/O
                self._set_vram_priority("LOW")
                try:
                    p.nice(psutil.HIGH_PRIORITY_CLASS)
                except:
                    p.nice(psutil.ABOVE_NORMAL_PRIORITY_CLASS)
                    
            elif intent_type == "DEEP_THOUGHT":
                # Compute intensive
                self._set_vram_priority("MAX")
                try:
                    p.nice(psutil.HIGH_PRIORITY_CLASS)
                except:
                    pass
            else:
                # Normal
                self._set_vram_priority("NORMAL")
                try:
                    p.nice(psutil.NORMAL_PRIORITY_CLASS)
                except:
                    pass
                    
        except ImportError:
            logger.warning("   psutil not found. Cannot set process priority.")
        except Exception as e:
            logger.error(f"  Failed to shift gears: {e}")

    def _set_vram_priority(self, level: str):
        logger.info(f"   [VRAM] -> {level}")

    def _set_io_priority(self, level: str):
        logger.info(f"   [NVMe] -> {level}")

    def get_metabolic_status(self) -> str:
        # Refresh the sensor data
        self.report = BodySensor.sense_body()
        self.strategy = self.report["strategy"]
        vessel = self.report.get("vessel", {})
        ram_pct = vessel.get("ram_percent", 0)
        cpu_pct = vessel.get("cpu_percent", 0)
        vram_gb = vessel.get("gpu_vram_total_gb", 0)
        
        # Narrative translation
        if ram_pct > 80:
            body_feel = "My mental stage is crowded (RAM High)."
        elif ram_pct < 30:
            body_feel = "My mind is clear and spacious (RAM Low)."
        else:
            body_feel = "My cognitive load is balanced."
            
        return f"{self.strategy} | CPU:{cpu_pct:.0f}% | RAM:{ram_pct:.0f}% | VRAM:{vram_gb:.1f}GB | {body_feel}"
