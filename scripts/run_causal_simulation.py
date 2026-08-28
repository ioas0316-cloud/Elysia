#!/usr/bin/env python3
"""
[Causal Simulation Demonstration Script]
Demonstrates the 3-Tier Causal Mechanics:
1. High-Level Intent Field / Engram Topology
2. Middle Cognition Field
3. Lower Physical Pathfinding

Scenario: Pokemon Trainer AI Agent walking Pallet Town -> Viridian City -> Pewter City.
Shows local infinite loop when Engram is inactive, and loop escape upon Engram exposure.
"""

import sys
import time
import numpy as np
from simulators.causal_grid_sim import CausalTilemapSimulator

def run_simulation_demo():
    print("=" * 70)
    print("      ELYSIUS CAUSAL FIELD SIMULATION DEMONSTRATION")
    print("   [Integrating High-Level Intent & Lower Physical Feedback Loop]")
    print("=" * 70)
    print("\nLandmarks:")
    print("  - Pallet Town: (0.0, 0.0)")
    print("  - Viridian City: (0.0, 10.0)")
    print("  - Pewter City: (0.0, 20.0)")
    print("-" * 70)

    sim = CausalTilemapSimulator()

    print("\n>>> PHASE 1: Running Agent without High-Level Engram Exposure...")
    print("    Expected Behavior: Local Minimum Habit Loop (Pallet Town <-> Viridian City)")
    print("-" * 70)

    for i in range(15):
        log = sim.step()
        pos_str = f"({log['position'][0]:5.2f}, {log['position'][1]:5.2f})"
        encounter_str = f" | {log['encounter']}" if log['encounter'] else ""
        print(f"Step {log['step']:02d} | Pos: {pos_str} | Landmark: {log['landmark']:13s} | Engram Force: {log['engram_force_norm']:.2f}{encounter_str}")

    print("\n>>> PHASE 2: Exposing High-Level Engram ('Oak's Parcel / Deliver to Pewter City')...")
    print("    Expected Behavior: Top-Down Causal Gradient alters potential topology, breaking infinite loop!")
    print("-" * 70)

    sim.set_engram_exposure(True)

    for i in range(15):
        log = sim.step()
        pos_str = f"({log['position'][0]:5.2f}, {log['position'][1]:5.2f})"
        encounter_str = f" | {log['encounter']}" if log['encounter'] else ""
        print(f"Step {log['step']:02d} | Pos: {pos_str} | Landmark: {log['landmark']:13s} | Engram Force: {log['engram_force_norm']:.2f}{encounter_str}")

    print("\n" + "=" * 70)
    print("                     SIMULATION SUMMARY")
    print("=" * 70)
    print(f"Total Steps Taken : {len(sim.history)}")
    print(f"Final Agent Position : ({sim.agent_voxel.position[0]:.2f}, {sim.agent_voxel.position[1]:.2f})")
    print(f"Reached Pewter City  : {np.linalg.norm(sim.agent_voxel.position - sim.locations['Pewter City']) < 1.5}")
    print(f"Total Encounters     : {len(sim.encounter_events)}")
    print("=" * 70)

if __name__ == "__main__":
    run_simulation_demo()
