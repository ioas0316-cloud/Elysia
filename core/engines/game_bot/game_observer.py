"""
Elysia Game-Bot State Observer
==============================
Connects to the persisted state file and displays realtime information 
about coordinate quaternions, sub-rotors, keyboard outputs, and collapse history.
"""

import os
import pickle

# We import the classes to ensure pickle can deserialize correctly
from engines.game_bot.game_engine import GameBotEngine, SubRotor
from core.math_utils import Quaternion

STATE_FILE = "elysia_state.pkl"
COLLAPSE_DIR = "collapses"

def load_state() -> GameBotEngine:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            print(f"Error loading state: {e}")
            return None
    return None

def format_quat(q: Quaternion) -> str:
    return f"[{q.w:.4f}, {q.x:.4f}, {q.y:.4f}, {q.z:.4f}]"

def observe():
    print("=====================================================================")
    print("   Elysia Observer Interface (Floating Axis / Azeroth Projection)")
    print("=====================================================================")
    
    state = load_state()
    if state is None:
        print("No active state found. Make sure the Game Bot Engine is running.")
        return

    print(f"[{state.name}] Realtime Diagnostics:")
    print(f"- Fractal Depth (Manifold): {state.fractal_depth}D")
    print(f"- Floating Axis Quat: {format_quat(state.internal_quat)}")
    print(f"- State Lock: {'Locked (0)' if state.is_locked else 'Unlocked/Chaos (1)'}")
    print(f"- Evolution Cycle: {state.cycle_count}")
    print(f"- Trajectory Memory: {len(state.trajectory_memory)} / {state.MAX_TRAJECTORY_LENGTH}")

    print(f"\n[Motor Cortex Status]")
    print(f"- Last Keyboard Action: {state.actuator.last_action}")

    print("\n[Visual Cortex Data]")
    rect = state.senser.get_capture_rect()
    if rect:
        print(f"- Observation Window: {rect['width']}x{rect['height']} px (Center-aligned)")
    else:
        print("- Observation Window: Uninitialized")

    print("\n[Sub-Rotors (Floating Axes)]:")
    for sr in state.sub_rotors:
        print(f"  Rotor {sr.id}: {format_quat(sr.quat)}")

    print("\n--- Recent Historical Events (Collapse & Death Logs) ---")
    if os.path.exists(COLLAPSE_DIR):
        events = [f for f in os.listdir(COLLAPSE_DIR) if f.endswith(".pkl")]
        if not events:
            print("No major collapse events or character deaths recorded yet.")
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
                            print(f"      Mismatch Angle: {e_data['mismatch']:.4f} rad | Weather Vector: {e_data['weather_vector']}")
                except Exception as ex:
                    print(f"  > Error reading {e}: {ex}")
    else:
        print("Events directory not found.")
    print("=====================================================================")

if __name__ == "__main__":
    observe()
