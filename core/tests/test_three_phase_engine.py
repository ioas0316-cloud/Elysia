import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from core.three_phase_four_wire_log_engine import ThreePhaseFourWireEngine

def run_simulation(scenario, kwargs):
    print(f"\n--- Running Scenario: {scenario} ---")
    engine = ThreePhaseFourWireEngine(bypass_threshold=3)
    try:
        # Run 3 perfect ticks to stabilize and enter bypass
        for _ in range(3):
            engine.process_tick(v_mag=220.0)

        # Run 1 tick in bypass mode
        engine.process_tick(v_mag=220.0)

        # Introduce anomaly causing relay break
        engine.process_tick(**kwargs)

    except SystemExit:
        print(f"--- Scenario {scenario} resulted in a Hard Crash (Surge) as expected. ---\n")
    except Exception as e:
        print(f"--- Scenario {scenario} encountered unexpected exception: {e} ---\n")

if __name__ == "__main__":
    # Scenario 1: Device 87 (Differential Leakage) Trip
    run_simulation("Leakage Detection (Device 87)", {"v_mag": 220.0, "i_leakage": 5.0})

    # Scenario 2: Device 59 (Over Voltage) Trip
    run_simulation("Over Voltage Surge (Device 59)", {"v_mag": 300.0, "current_perturbation": 0.0001})

    # Scenario 3: Device 27 (Under Voltage) Trip
    run_simulation("Under Voltage Surge (Device 27)", {"v_mag": 150.0, "current_perturbation": 0.0001})

    # Scenario 4: Device 78 (Phase out of step / Hallucination) Trip
    run_simulation("Out of Step / Hallucination (Device 78)", {"v_mag": 220.0, "phase_shift_noise": 0.5})

    # Scenario 5: KCL Imbalance (N Phase)
    run_simulation("KCL Imbalance (Device 21 / 25 / N Phase)", {"v_mag": 220.0, "current_perturbation": 10.0})
