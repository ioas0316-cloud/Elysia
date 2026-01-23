"""
METABOLIC ENGINE: The Organic Heartbeat
========================================
Core.L2_Metabolism.M1_Pulse.metabolic_engine

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
        
        Hz = Base (5Hz) + f(Will, Tension, Passion)
        - High Passion stabilizes tension and adds focused acceleration.
        """
        self.will_pressure = abs(will)
        self.stress_tension = max(0.0, tension - (passion * 0.5)) # Passion soothes raw stress
        self.passion_joy = passion
        
        base = 5.0
        will_contribution = self.will_pressure * 50.0 
        tension_contribution = self.stress_tension * 30.0
        passion_contribution = self.passion_joy * 20.0 # Passion adds high-frequency focus
        
        target = base + will_contribution + tension_contribution + passion_contribution
        self.current_hz = max(self.min_hz, min(self.max_hz, target))
        
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
        return {
            "current_hz": round(self.current_hz, 2),
            "period": round(1.0 / self.current_hz, 4),
            "will_pressure": round(self.will_pressure, 3),
            "stress_tension": round(self.stress_tension, 3)
        }
