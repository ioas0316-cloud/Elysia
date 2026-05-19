"""
Hybrid Drive Demo: Reflex vs. Thought
=====================================
Core.Demos.hybrid_demo

Demonstrates the "Zero-Computation" advantage of Rotors vs. Thundercloud.
"""

import time
import numpy as np
from Core.System.rotor import Rotor, RotorConfig
from Core.Divine.muscle_memory import MuscleMemory
from Core.System.Merkaba.thundercloud import Thundercloud
from Core.Divine.monad_core import Monad
from Core.System.double_helix_dna import DoubleHelixDNA

def run_simulation():
    print("⚡ HYBRID DRIVE SIMULATION ⚡")
    print("===========================")

    # 1. Setup Systems
    # A. Cerebellum (Muscle Memory)
    cerebellum = MuscleMemory()
    face_rotor = Rotor("Face.Smile", RotorConfig(idle_rpm=0))
    cerebellum.register_rotor(face_rotor)

    # Bake a "Smile" animation (Simple Curve 0 -> 90 -> 0)
    smile_track = [0, 10, 30, 60, 90, 90, 90, 60, 30, 10, 0]
    cerebellum.learn_reflex("Greeting", "Smile", smile_track, "Face.Smile")

    # B. Cerebrum (Thundercloud)
    cloud = Thundercloud()
    # Create a heavy concept to simulate load
    complex_dna = DoubleHelixDNA(
        pattern_strand=np.random.randn(1024).astype(np.float32),
        principle_strand=np.random.randn(7).astype(np.float32)
    )
    cloud.active_monads = [Monad(f"Thought_{i}", dna=complex_dna) for i in range(100)]
    for m in cloud.active_monads:
        cloud._monad_map[m.seed] = m

    # 2. Test 1: Reflex Action (Greet)
    print("\n[TEST 1] Intent: 'Greeting'")
    start_time = time.perf_counter()

    # Try Reflex First
    triggered = cerebellum.try_reflex("Greeting")

    end_time = time.perf_counter()
    duration = (end_time - start_time) * 1000 # ms

    if triggered:
        print(f">>> REFLEX TRIGGERED: 'Smile' (Cost: {duration:.4f} ms)")
        print(f"    Rotor State: {face_rotor.active_track} (Playing: {face_rotor.is_playing})")

        # Simulate Playback
        print("    Playing: ", end="")
        while face_rotor.is_playing:
            face_rotor.update(0.16)
            print(f"{face_rotor.current_angle:.0f}° ", end="", flush=True)
        print("\n    Action Complete.")
    else:
        print(">>> REFLEX FAILED. Fallback to Cloud.")

    # 3. Test 2: Complex Thought (Analyze)
    print("\n[TEST 2] Intent: 'Analyze'")
    start_time = time.perf_counter()

    # Try Reflex
    triggered = cerebellum.try_reflex("Analyze")

    if not triggered:
        # Fallback to Thundercloud
        cloud.ignite("Thought_0", voltage=1.0)

        end_time = time.perf_counter()
        duration = (end_time - start_time) * 1000 # ms
        print(f">>> THUNDERCLOUD IGNITED (Cost: {duration:.4f} ms)")
        print("    (Physics simulation executed)")

if __name__ == "__main__":
    run_simulation()
