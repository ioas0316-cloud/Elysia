import sys
import os
import cmath
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from core.elysia_hyper_rotor_engine import HyperRotorEngine

def run_tests():
    print("=== SCENARIO 1: DIMENSIONAL EXPANSION (ASSIMILATING UNKNOWN NOISE) ===")
    engine1 = HyperRotorEngine()

    # Tick 1: Normal Wye Operation
    engine1.process_tick(voltage_input=220.0, external_noise_phasor=0j)

    # Tick 2: Introduce an unknown 4th dimensional noise (e.g. at 45 degrees, 50 magnitude)
    # Slow introduction to not trip the Delta-Y velocity shift (dtheta/dt < 15 deg)
    # We will introduce it slowly to test pure dimensional expansion
    noise_phasor = cmath.rect(50.0, math.radians(0)) # Velocity 0
    engine1.process_tick(voltage_input=220.0, external_noise_phasor=noise_phasor)

    # Tick 3: The system should have generated Axis D to counter it.
    engine1.process_tick(voltage_input=220.0, external_noise_phasor=noise_phasor)

    # Tick 4: Verify stability with 4 axes
    engine1.process_tick(voltage_input=220.0, external_noise_phasor=noise_phasor)

    print("\n\n=== SCENARIO 2: DOUBLE-HELIX FUTURE PREDICTION & DELTA-Y SHIFT ===")
    engine2 = HyperRotorEngine()

    # Tick 1: Normal
    engine2.process_tick(voltage_input=220.0)

    # Tick 2: Introduce a sudden massive phase shift (simulating acceleration/instability)
    # This will cause a large dtheta/dt
    engine2.process_tick(voltage_input=220.0, external_noise_phasor=cmath.rect(20.0, math.radians(90)))

    # Tick 3: The velocity should be high, triggering Delta shift
    engine2.process_tick(voltage_input=220.0, external_noise_phasor=cmath.rect(40.0, math.radians(180)))

    # Tick 4: In Delta, the system stabilizes the velocity
    engine2.process_tick(voltage_input=220.0, external_noise_phasor=cmath.rect(40.0, math.radians(180)))

    # Tick 5: Shifting back to Wye
    engine2.process_tick(voltage_input=220.0, external_noise_phasor=cmath.rect(40.0, math.radians(180)))

if __name__ == "__main__":
    run_tests()
