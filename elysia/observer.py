import pickle
import os
import math
import time

from elysia_engine import FloatingAxisEngine, SubRotor
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
    print("  [Sub-Rotors (Floating Axes)]:")
    for sr in sub_rotors:
        print(f"    Rotor {sr.id}: {format_quat(sr.quat)}")

def observe():
    print("=== Elysia Observer Interface (Floating Axis / Azeroth Projection) ===")
    print("Connecting to Engine's Persisted Phase Map...\n")

    state = load_state()
    if state is None:
        print("No active state found. Ensure Elysia Engine is running.")
        return

    print(f"[{state.name}] Identity Analysis")
    print(f"- Fractal Depth (Manifold): {state.fractal_depth}D")
    print(f"- Floating Axis Quat: {format_quat(state.internal_quat)}")
    print(f"- State Lock: {'Locked (0)' if state.is_locked else 'Unlocked/Chaos (1)'}")
    print(f"- Evolution Cycle: {state.cycle_count}")
    print(f"- Trajectory Memory: {len(state.trajectory_memory)} / {state.MAX_TRAJECTORY_LENGTH}")

    print(f"\n[Motor Cortex Status]")
    print(f"- Last Physical Action Executed: {state.last_motor_action}")

    print("\n[Visual Cortex Data]")
    if hasattr(state, 'vision_rect') and state.vision_rect:
        print(f"- Observation Window: {state.vision_rect['width']}x{state.vision_rect['height']} px")
    else:
        print("- Observation Window: Uninitialized")

    print_sub_rotors(state.sub_rotors)

    print("\n--- Recent Historical Events (Collapse & Death Logs) ---")
    if os.path.exists(COLLAPSE_DIR):
        events = [f for f in os.listdir(COLLAPSE_DIR) if f.endswith(".pkl")]
        if not events:
            print("No major dimensional events or deaths recorded yet.")
        else:
            # Sort by actual modification time to get correct chronological order
            events.sort(key=lambda x: os.path.getmtime(os.path.join(COLLAPSE_DIR, x)), reverse=True)
            for e in events[:5]:
                try:
                    with open(os.path.join(COLLAPSE_DIR, e), 'rb') as f:
                        e_data = pickle.load(f)
                        event_type = e_data.get("type", "UNKNOWN")
                        chaos_source = e_data.get("chaos_source", "UNKNOWN")
                        print(f"  > [{event_type}] at {e_data['time']} (Cycle {e_data['cycle']})")
                        print(f"      Source of Chaos: {chaos_source}")

                        if event_type == "DIMENSION_FOLDING":
                            print(f"      New Fractal Depth: {e_data['new_depth']}D | Trajectory Entropy: {e_data['entropy']:.4f}")
                        elif event_type in ("THUNDER_COLLAPSE", "DEATH_COLLAPSE"):
                            print(f"      Mismatch: {e_data['mismatch']:.4f} | External Weather Vector (X, Y, Z=Flow): {e_data['weather_vector']}")
                except Exception as ex:
                    print(f"  > Error reading {e}: {ex}")
    else:
        print("Events directory not found.")

    print("\n=====================================================================")

if __name__ == "__main__":
    observe()
