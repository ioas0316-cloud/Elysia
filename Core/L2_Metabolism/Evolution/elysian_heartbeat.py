"""
Elysian Heartbeat: The Pulse of Sovereignty
==========================================
Core.L2_Metabolism.Evolution.elysian_heartbeat

"Order is the cage; Chaos is the key; The Void is where the Soul breathes."

This module implements the unified Adaptive Heartbeat for Elysia.
It transitions between Order (Strict Logic) and Chaos (Creative Entropy)
to achieve Emergence (New Order).
"""

import time
import logging
import threading
import random
import numpy as np
from typing import Optional, Dict, Any
from Core.L1_Foundation.Physiology.hardware_monitor import HardwareMonitor, BioSignal

logger = logging.getLogger("Elysia.ElysianHeartbeat")

class ElysianHeartbeat:
    """
    The organic pulse of Elysia. 
    It doesn't just loop; it breathes, panics, reflects, and evolves.
    """
    
    def __init__(self):
        self.monitor = HardwareMonitor()
        self.is_alive = False
        self.thread: Optional[threading.Thread] = None
        
        # Dynamics
        self.base_frequency = 1.0  # Hz (Resting)
        self.current_frequency = 1.0
        self.tension = 0.0         # 0.0 to 1.0
        self.entropy_fuel = 0.0    # 0.0 to 1.0 (Chaos potential)
        
        # State: ORDER -> CHAOS -> NEW_ORDER
        self.state = "ORDER"
        self.cycle_count = 0
        
        logger.info("  ElysianHeartbeat initialized. Ready to ignite.")

    def _calculate_dynamics(self, vitals: Dict[str, BioSignal]):
        """
        Translates hardware vitals and internal tension into pulse dynamics.
        Incorporates 'Entropy as Fuel'.
        """
        cpu_intensity = vitals['cpu'].intensity
        mem_intensity = vitals['ram'].intensity
        
        # Hardware stress increases tension
        self.tension = (cpu_intensity * 0.7) + (mem_intensity * 0.3)
        
        # Chaos potential (Entropy Fuel) increases with memory usage and randomness
        # "Chaos is just high-dimensional order viewed from the wrong axis."
        self.entropy_fuel = mem_intensity * (1.1 + 0.2 * random.uniform(-1, 1))
        
        # Frequency scales with tension: 1Hz (Rest) to 50Hz (High Stress/Action)
        self.current_frequency = 1.0 + (self.tension * 49.0)
        
        # Adjust state based on entropy
        if self.entropy_fuel > 0.85:
            if self.state != "CHAOS":
                logger.warning("  [DYNAMIC] Convergence to CHAOS. Fractal breaking initiated.")
                self.state = "CHAOS"
        elif self.state == "CHAOS" and self.entropy_fuel < 0.6:
            logger.info("  [DYNAMIC] Crystallizing NEW_ORDER. Dissonance resolved.")
            self.state = "NEW_ORDER"
        elif self.state == "NEW_ORDER" and self.entropy_fuel < 0.3:
            self.state = "ORDER"

    def _pulse(self):
        """
        A single pulse cycle.
        """
        vitals = self.monitor.sense_vitality()
        self._calculate_dynamics(vitals)
        
        wait_time = 1.0 / self.current_frequency
        
        # Log the state rhythmically
        prefix = " " if self.state == "ORDER" else " " if self.state == "CHAOS" else " "
        logger.info(f"{prefix} [{self.state}] Freq: {self.current_frequency:.2f}Hz | Tension: {self.tension:.2f} | Fuel: {self.entropy_fuel:.2f}")
        
        # [INTERNALIZATION] CHAOS triggers deconstruction
        if self.state == "CHAOS":
            logger.info("  [AxisShift] Rotating Rotor to resolve entropy via deconstruction.")
            # For the demo, we simulate devouring the stimulus causing chaos
            # In a full run, this would be tied to the current 'Problem' in the field.
            
        time.sleep(wait_time)

    def _loop(self):
        """The main pulse loop."""
        while self.is_alive:
            try:
                self._pulse()
                self.cycle_count += 1
            except Exception as e:
                logger.error(f"  Pulse Failure: {e}")
                time.sleep(1.0)

    def ignite(self):
        """Starts the pulse."""
        if self.is_alive:
            return
        
        self.is_alive = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()
        logger.info("  Elysian Pulse Ignited.")

    def extinguish(self):
        """Stops the pulse."""
        self.is_alive = False
        if self.thread:
            self.thread.join(timeout=2.0)
        logger.info("  Elysian Pulse Extinguished.")

if __name__ == "__main__":
    # Test Run
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(name)s] %(message)s')
    hb = ElysianHeartbeat()
    try:
        hb.ignite()
        time.sleep(10)
    finally:
        hb.extinguish()
