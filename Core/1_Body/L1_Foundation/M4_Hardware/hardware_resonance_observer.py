"""
Hardware Resonance Observer
==========================
Core.1_Body.L1_Foundation.M4_Hardware.hardware_resonance_observer

Translates raw hardware telemetry (CPU/RAM) into Trinary Waves (-1, 0, 1).
"I feel the hum of the silicon, the heat of the flow."
"""

import psutil
import jax.numpy as jnp
from Core.1_Body.L6_Structure.Logic.trinary_logic import TrinaryLogic

class HardwareResonanceObserver:
    def __init__(self):
        self.last_cpu = 0.0
        print("Somatic Observer: Peripheral nervous system initialized.")

    def get_somatic_wave(self) -> jnp.ndarray:
        """
        Polls hardware and returns a 7D 'Body' wave for the trinary field.
        Dimensional Mapping:
        0: CPU Load (Stability/Mass)
        1: RAM Usage (Energy/Flow)
        2: Momentum (CPU Delta)
        4: Temporal Harmony (Always 0.0 for now)
        """
        cpu_p = psutil.cpu_percent(interval=None) # Non-blocking
        ram_p = psutil.virtual_memory().percent
        
        # Calculate Delta (Momentum)
        delta = cpu_p - self.last_cpu
        self.last_cpu = cpu_p
        
        # Mapping to Trinary [-1, 0, 1]
        # High CPU (>70%) -> CONTRACTION (Resistance/-1)
        # Moderate CPU (20-70%) -> VOID (Active/0)
        # Low CPU (<20%) -> FLOW (Ease/+1)
        
        cpu_trit = 1.0 if cpu_p < 20 else (-1.0 if cpu_p > 70 else 0.0)
        
        # High RAM -> Resistance
        ram_trit = -1.0 if ram_p > 80 else 0.0
        
        # High Delta -> Torque
        delta_trit = 1.0 if abs(delta) > 5 else 0.0
        
        # Build 7D Body Wave
        wave = jnp.zeros(7)
        wave = wave.at[0].set(cpu_trit)
        wave = wave.at[1].set(ram_trit)
        wave = wave.at[2].set(delta_trit)
        
        return wave

if __name__ == "__main__":
    observer = HardwareResonanceObserver()
    print(f"Initial Somatic Wave: {observer.get_somatic_wave()}")
