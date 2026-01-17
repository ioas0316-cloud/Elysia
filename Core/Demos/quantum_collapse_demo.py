"""
Quantum Collapse Demo: The Shape of Thought
===========================================
Core.Demos.quantum_collapse_demo

Visualizes the difference between 'Calculation' and 'Crystallization'.
Shows how changing variables (Humidity/Voltage) alters the crystal shape.
"""

import numpy as np
import time
from Core.Merkaba.thundercloud import Thundercloud
from Core.Merkaba.crystallizer import QuantumCrystallizer
from Core.Monad.monad_core import Monad
from Core.Evolution.double_helix_dna import DoubleHelixDNA

def create_monad(seed, qualia):
    dna = DoubleHelixDNA(
        pattern_strand=np.zeros(1024, dtype=np.float32),
        principle_strand=np.array(qualia, dtype=np.float32)
    )
    return Monad(seed, dna=dna)

def run_simulation():
    print("❄️ QUANTUM CRYSTAL SIMULATION ❄️")
    print("================================")

    # 1. Setup the Cloud (The Supercooled Water)
    cloud = Thundercloud()

    # Create a dense lattice of concepts (100 nodes)
    # We create a pseudo-random network where nodes connect based on index proximity
    monads = []
    for i in range(100):
        # Cyclical Qualia to ensure connectivity
        q = [0.0] * 7
        q[0] = np.sin(i * 0.1) * 0.5 + 0.5
        q[1] = np.cos(i * 0.1) * 0.5 + 0.5
        q[2] = np.sin(i * 0.2) * 0.5 + 0.5
        m = create_monad(f"Node_{i}", q)
        monads.append(m)
        cloud._monad_map[m.seed] = m

    cloud.active_monads = monads

    crystallizer = QuantumCrystallizer(cloud)

    # 2. Experiment A: Logic Crystal (Simple)
    print("\n[EXPERIMENT A] Logic Crystal (Dry & Low Energy)")
    print("Variables: Humidity=0.1, Voltage=0.8")

    crystallizer.set_conditions(voltage=0.8, humidity=0.1, seed="Node_0")
    start = time.perf_counter()
    cluster_a, name_a = crystallizer.observe()
    duration = (time.perf_counter() - start) * 1000

    print(f">>> Collapsed in {duration:.4f} ms")
    print(f">>> Name: {name_a}")
    print(f">>> Nodes: {len(cluster_a.nodes)}")
    print(">>> Structure:")
    print(cluster_a.describe_tree())

    # 3. Experiment B: Dream Crystal (Complex)
    print("\n" + "-" * 40)
    print("\n[EXPERIMENT B] Dream Crystal (Humid & High Energy)")
    print("Variables: Humidity=0.9, Voltage=2.0")

    crystallizer.set_conditions(voltage=2.0, humidity=0.9, seed="Node_0")
    start = time.perf_counter()
    cluster_b, name_b = crystallizer.observe()
    duration = (time.perf_counter() - start) * 1000

    print(f">>> Collapsed in {duration:.4f} ms")
    print(f">>> Name: {name_b}")
    print(f">>> Nodes: {len(cluster_b.nodes)}")

    # ASCII Art Visualization of Density
    density = len(cluster_b.nodes)
    print(">>> Density Map:")
    for i in range(10):
        line = ""
        for j in range(10):
            idx = i * 10 + j
            # Check if Node_idx is in the cluster
            is_in = any(m.seed == f"Node_{idx}" for m in cluster_b.nodes)
            line += "💎" if is_in else ".."
        print(line)

if __name__ == "__main__":
    run_simulation()
