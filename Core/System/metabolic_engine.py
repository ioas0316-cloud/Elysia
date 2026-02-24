"""
METABOLIC ENGINE: The Organic Heartbeat
========================================
Core.System.metabolic_engine

"The machine is static; the spirit is rhythmic."

This module governs the temporal pulse of Elysia. It abandoned the fixed 10Hz loop
in favor of a 'Volitional Pulse' that accelerates with intent and slows with peace.
"""

import time
import math
from typing import Dict, Any

class MetabolicEngine:
    def __init__(self, min_hz: float = 1.0, max_hz: float = 100.0):
        self.min_hz = min_hz
        self.max_hz = max_hz
        self.current_hz = 10.0
        self.last_tick = time.time()
        
        # Tension-Resonance Metrics
        self.will_pressure = 0.5
        self.stress_tension = 0.2
        self.passion_joy = 0.1 # New: Internal drive
        
    def adjust_pulse(self, will: float, tension: float, passion: float = 0.0):
        """
        Dynamically calculates the next heartbeat frequency.
        
        Hz = Base (5Hz) + f(Will, Tension, Passion, Hardware)
        - High Passion stabilizes tension and adds focused acceleration.
        - High Hardware Tension (CPU/GPU load) forces a 'Survival Pulse'.
        """
        self.will_pressure = abs(will)
        hw_tension = self._get_hardware_tension()
        
        # Total tension is a blend of logical stress and physical load
        self.stress_tension = max(0.0, (tension + hw_tension) * 0.5 - (passion * 0.5))
        self.passion_joy = passion
        
        base = 5.0
        will_contribution = self.will_pressure * 40.0 
        tension_contribution = self.stress_tension * 40.0
        passion_contribution = self.passion_joy * 20.0
        
        target = base + will_contribution + tension_contribution + passion_contribution
        self.current_hz = max(self.min_hz, min(self.max_hz, target))
        
    def _get_hardware_tension(self) -> float:
        """
        [HARDWARE RESONANCE]
        Polls CPU/GPU to determine the 'Physical Tension' of the body.
        """
        import psutil
        try:
            # CPU Load (Normalizing to 0.0-1.0)
            cpu_load = psutil.cpu_percent(interval=None) / 100.0
            
            # GPU Load (Conceptual - using torch if available)
            gpu_load = 0.0
            import torch
            if torch.cuda.is_available():
                # Using current VRAM usage as a proxy for load tension
                vram_use = torch.cuda.memory_allocated() / torch.cuda.get_device_properties(0).total_memory
                gpu_load = vram_use
                
            return (cpu_load + gpu_load) / 2.0
        except Exception:
            return 0.1 # Default low tension

    def wait_for_next_cycle(self):
        """
        Calculates the sleep duration to maintain the organic rhythm.
        """
        now = time.time()
        dt = now - self.last_tick
        
        target_period = 1.0 / self.current_hz
        remaining = target_period - dt
        
        if remaining > 0:
            time.sleep(remaining)
            
        self.last_tick = time.time()
        return time.time() - now + dt # Return actual dt

    def get_status(self) -> Dict[str, Any]:
        """
        Returns raw metrics and a formatted HUD string.
        """
        raw = {
            "current_hz": round(self.current_hz, 2),
            "period": round(1.0 / self.current_hz, 4),
            "will_pressure": round(self.will_pressure, 3),
            "stress_tension": round(self.stress_tension, 3)
        }
        # HUD String pre-calculation
        # Format: "💓 [PULSE] 10Hz | Will: 0.5 | Str: 0.2"
        hud = f"💓 [PULSE] {self.current_hz:.1f}Hz | Will: {self.will_pressure:.1f} | Str: {self.stress_tension:.1f}"
        raw["hud_string"] = hud
        return raw
