import sys
import os
import cmath
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from core.elysia_hyper_rotor_engine import HyperRotorEngine

def run_tests():
    print("=== SCENARIO 1: HOMEOSTASIS (Bypass -> Anomaly -> Expansion -> Bypass) ===")
    engine1 = HyperRotorEngine(bypass_threshold=2)

    # Ticks 1-3: Establish Bypass Mode
    engine1.process_tick()
    engine1.process_tick()
    engine1.process_tick() # Should enter bypass

    # Tick 4: In bypass
    engine1.process_tick()

    # Tick 5: Suddenly, an anomaly hits! Break Bypass and spawn Axis D
    noise = cmath.rect(50.0, math.radians(10))
    engine1.process_tick(external_noise_phasor=noise)

    # Tick 6: Engine assimilates the noise. I_N drops to 0.
    engine1.process_tick(external_noise_phasor=noise)

    # Ticks 7-8: Stabilizing... Returns to Bypass
    engine1.process_tick(external_noise_phasor=noise)
    engine1.process_tick(external_noise_phasor=noise)

    print("\n\n=== SCENARIO 2: CRITICAL SURGE (No Hard Crash) ===")
    engine2 = HyperRotorEngine()

    # Tick 1: Normal
    engine2.process_tick()

    # Tick 2: Massive surge > 100A
    massive_surge = cmath.rect(150.0, math.radians(0))
    engine2.process_tick(external_noise_phasor=massive_surge)

    # Tick 3: System is still spinning and evaluating, but flagged SURGE_CRITICAL
    engine2.process_tick(external_noise_phasor=massive_surge)


if __name__ == "__main__":
    run_tests()
