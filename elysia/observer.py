import pickle
import os
import math
import time

from elysia_engine import RecursiveUnit, SubRotor
from pyquaternion import Quaternion

STATE_FILE = "elysia_state.pkl"
COLLAPSE_DIR = "collapses"

def load_state():
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading state: {e}")
            return None
    return None

def format_quat(q):
    return f"[{q.w:.4f}, {q.x:.4f}, {q.y:.4f}, {q.z:.4f}]"

def print_sub_rotors(sub_rotors):
    print("  [Sub-Rotors (3D Orientation)]:")
    for sr in sub_rotors:
        print(f"    Rotor {sr.id}: {format_quat(sr.quat)}")

def observe():
    print("=== Elysia Observer Interface (Multi-Dimensional) ===")
    print("Connecting to Engine's Persisted Phase Map...\n")

    state = load_state()
    if state is None:
        print("No active state found. Ensure Elysia Engine is running.")
        return

    print(f"[{state.name}] Identity Analysis")
    print(f"- Fractal Depth: {state.fractal_depth}D")
    print(f"- Current Internal Quat: {format_quat(state.internal_quat)}")
    print(f"- State Lock: {'Locked (0)' if state.is_locked else 'Unlocked/Chaos (1)'}")
    print(f"- Evolution Cycle: {state.cycle_count}")
    print(f"- Trajectory Memory Size: {len(state.trajectory_memory)} / {state.MAX_TRAJECTORY_LENGTH}")

    print_sub_rotors(state.sub_rotors)

    print("\n--- Recent Historical Events ---")
    if os.path.exists(COLLAPSE_DIR):
        events = [f for f in os.listdir(COLLAPSE_DIR) if f.endswith(".pkl")]
        if not events:
            print("No major dimensional events recorded yet.")
        else:
            events.sort(reverse=True) # Show latest first
            for e in events[:5]:
                try:
                    with open(os.path.join(COLLAPSE_DIR, e), 'rb') as f:
                        e_data = pickle.load(f)
                        event_type = e_data.get("type", "UNKNOWN")
                        print(f"  > [{event_type}] at {e_data['time']} (Cycle {e_data['cycle']})")

                        if event_type == "DIMENSION_FOLDING":
                            print(f"      New Fractal Depth: {e_data['new_depth']}D | Trajectory Entropy: {e_data['entropy']:.4f}")
                        elif event_type == "THUNDER_COLLAPSE":
                            print(f"      Mismatch: {e_data['mismatch']:.4f} | External Weather Vector (X,Y,Z): {e_data['weather_vector']}")
                except Exception as ex:
                    print(f"  > Error reading {e}: {ex}")
    else:
        print("Events directory not found.")

    print("\n=====================================================")

if __name__ == "__main__":
    observe()
