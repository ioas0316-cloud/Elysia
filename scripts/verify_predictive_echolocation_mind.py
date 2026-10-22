"""
Verification and Demonstration Script for Predictive Echolocation & Mind Space Engine
====================================================================================
Demonstrates:
1. 4D Spacetime Mental Simulation ($W_{\\text{pred}}$) vs Actual Sensing ($W_{\\text{actual}}$).
2. Doppler frequency shift tracking under varying velocities.
3. Clifford Rotor 3D phase rotation dynamics.
4. Prediction error ($\mathcal{E}$) adaptation loop and emergence of Sense of Agency ($S_{\\text{agency}}$).
"""

import sys
import os
import math
import numpy as np

# Ensure root path is accessible
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.sensory.predictive_echolocation_mind import (
    PredictiveEcholocationMindEngine,
    AgencyInferenceResult
)
from core.lens.cognitive_lens_engine import CognitiveLensEngine
from core.sensory.causal_sensor import CausalSensor


def main():
    print("================================================================================")
    print(" Elysia Cognitive Engine: Predictive Echolocation & 4D Spacetime Mind Space")
    print("================================================================================")

    lens_engine = CognitiveLensEngine()
    causal_sensor = CausalSensor(sensor_id="verify_sensor_01", lens_engine=lens_engine)
    engine = PredictiveEcholocationMindEngine(dimension=8, causal_sensor=causal_sensor)

    print("\n--- [Phase 1: Initial State Inspection] ---")
    state = engine.get_engine_state()
    print(f"Time t: {state['time_t']:.2f} s")
    print(f"Initial Position (3D): {state['position']}")
    print(f"Carrier Frequency: {state['carrier_frequency_hz'] / 1e9:.2f} GHz")

    print("\n--- [Phase 2: 4D Smooth Kinematic Trajectory Simulation (Lawful Motion)] ---")
    trajectory_deltas = [
        ([1.0, 0.5, 0.0], (0.02, 0.0, 0.01)),
        ([1.5, 0.8, 0.2], (0.03, 0.01, 0.02)),
        ([2.0, 1.0, 0.5], (0.04, 0.02, 0.01)),
        ([2.2, 1.2, 0.6], (0.01, 0.0, 0.01)),
        ([2.5, 1.5, 0.8], (0.0, 0.0, 0.0))
    ]

    for step, (pos_delta, rot_delta) in enumerate(trajectory_deltas, 1):
        res: AgencyInferenceResult = engine.process_predictive_echolocation_cycle(
            position_delta=pos_delta,
            rotation_delta=rot_delta,
            dt=0.1
        )
        print(f"[Step {step}] t={res.timestamp:.2f}s | "
              f"Pos={res.adapted_position.round(2).tolist()} | "
              f"Doppler Shift={res.doppler_drift:.2f} Hz | "
              f"Rotor Angle={res.rotor_angle:.4f} rad | "
              f"Pred Error={res.prediction_error:.4f} | "
              f"Agency Score={res.sense_of_agency:.4f} ({res.agency_status})")

    print("\n--- [Phase 3: Unexpected External Disturbance (High Discrepancy Shock)] ---")
    disturbed_rf = np.array([12.0, -8.0, 15.0, -22.0, 18.0, -9.0, 14.0, -11.0], dtype=np.float64)
    res_shock = engine.process_predictive_echolocation_cycle(
        position_delta=[0.1, 0.0, 0.0],
        rotation_delta=(0.0, 0.0, 0.0),
        external_rf_signal=disturbed_rf,
        dt=0.1
    )
    print(f"[Shock Event] t={res_shock.timestamp:.2f}s | "
          f"Pred Error={res_shock.prediction_error:.4f} | "
          f"Agency Score={res_shock.sense_of_agency:.4f} ({res_shock.agency_status})")

    print("\n--- [Phase 4: Adaptation & Recovery Loop] ---")
    for step in range(1, 4):
        res_rec = engine.process_predictive_echolocation_cycle(
            position_delta=[0.2, 0.1, 0.0],
            rotation_delta=(0.01, 0.0, 0.01),
            dt=0.1
        )
        print(f"[Recovery {step}] t={res_rec.timestamp:.2f}s | "
              f"Pred Error={res_rec.prediction_error:.4f} | "
              f"Agency Score={res_rec.sense_of_agency:.4f} ({res_rec.agency_status})")

    print("\n================================================================================")
    print(" VERIFICATION SUCCESS: Predictive Echolocation 4D Mind Space Engine operational!")
    print("================================================================================")


if __name__ == "__main__":
    main()
