#!/usr/bin/env python3
"""
Demo: Powder World & Spatiotemporal Causal Field Weaving
========================================================
Demonstrates:
1. Microscopic Continuous Powder World (Earth, Fire, Water, Gravity) potential field simulation.
2. Macroscopic 4D Spatiotemporal Causal Manifold Weaving ($W(x, y, layer, t)$).
3. O(1) Time-Slider ($t \\in [0, T]$) reconstruction: freely sliding backwards (retrocausal trace)
   and forwards in causal time without per-frame physics re-computations.
4. Rich ASCII/ANSI Grid Field Visualization of the woven ecosystem.
"""

import time
import numpy as np
from core.physics.causal_powder_field_weaving import PowderWorldCausalEngine

# ANSI Colors for rich grid visualization
RESET = "\033[0m"
BOLD = "\033[1m"
BROWN = "\033[38;5;130m"     # Earth / Powder
RED = "\033[38;5;196m"       # Thermal / Fire
BLUE = "\033[38;5;39m"       # Fluid / Water
PURPLE = "\033[38;5;129m"     # Fire + Water / Vapor
GREEN = "\033[38;5;46m"      # Organic / Settled
GRAY = "\033[38;5;240m"      # Empty / Grid


def render_ascii_field(state: dict, title: str, timestamp: float):
    """Renders 2D field state as a rich ASCII grid."""
    earth = state["earth"]
    thermal = state["thermal"]
    fluid = state["fluid"]
    height, width = earth.shape

    print(f"\n{BOLD}{'=' * 60}{RESET}")
    print(f"{BOLD}{title} [Causal Time t = {timestamp:.2f}s]{RESET}")
    print(f"{BOLD}{'=' * 60}{RESET}\n")

    grid_lines = []
    for y in range(height):
        row = []
        for x in range(width):
            e = earth[y, x]
            t = thermal[y, x]
            f = fluid[y, x]

            if t > 0.8 and f > 0.3:
                char = f"{PURPLE}♨{RESET}"  # Hot steam/vapor
            elif t > 0.5:
                char = f"{RED}▲{RESET}"     # Fire / High Heat
            elif f > 0.5:
                char = f"{BLUE}≈{RESET}"     # Water / Fluid
            elif e > 0.8:
                char = f"{BROWN}█{RESET}"     # Dense Earth
            elif e > 0.2:
                char = f"{BROWN}░{RESET}"     # Light Powder
            else:
                char = f"{GRAY}·{RESET}"     # Empty Void
            row.append(char)
        grid_lines.append(" ".join(row))

    print("\n".join(grid_lines))
    print(f"\n{GRAY}Legend: {BROWN}░█ Earth {RESET}| {RED}▲ Fire {RESET}| {BLUE}≈ Water {RESET}| {PURPLE}♨ Vapor {RESET}| {GRAY}· Void{RESET}\n")


def main():
    print(f"\n{BOLD}{'=' * 80}{RESET}")
    print(f"{BOLD} PROJECT ELYSIA :: POWDER WORLD & SPATIOTEMPORAL CAUSAL FIELD WEAVING {RESET}")
    print(f"{BOLD}{'=' * 80}{RESET}\n")

    # 1. Initialize Engine
    width, height = 16, 12
    engine = PowderWorldCausalEngine(width=width, height=height)
    print(f"-> Initialized 2D Powder Potential Field Grid ({width}x{height}).")

    # 2. Seed Initial Elements
    print("-> Seeding initial elementary powder heap, top water source, and bottom thermal source...")
    engine.seed_initial_powder_world()

    initial_state = engine.slide_time(0.0)
    render_ascii_field(initial_state, "PHASE 0: Initial World Seeding", 0.0)

    # 3. Execute Microscopic Simulation Phase
    simulation_duration = 3.0
    ticks = 30
    dt = simulation_duration / ticks
    print(f"-> Evolving microscopic field over {simulation_duration}s ({ticks} ticks)...")

    engine.run_simulation_phase(ticks=ticks, dt=dt)
    print(f"-> [WEAVING COMPLETE] 4D Causal Manifold Tensor W(x, y, layer, t) successfully woven!")

    final_state = engine.slide_time(simulation_duration)
    render_ascii_field(final_state, f"PHASE 1: Micro-Simulation Peak", simulation_duration)

    # 4. Demonstrate O(1) Time-Slider (Retrocausal Back-Tracing & Forward Sliding)
    print(f"\n{BOLD}{'=' * 80}{RESET}")
    print(f"{BOLD} DEMONSTRATING O(1) SPATIOTEMPORAL TIME-SLIDER (NO PHYSICS RE-COMPUTATIONS) {RESET}")
    print(f"{BOLD}{'=' * 80}{RESET}")

    slider_targets = [2.25, 1.50, 0.75, 0.00, 1.80, 3.00]
    for target_t in slider_targets:
        time.sleep(0.3)
        reconstructed_state = engine.slide_time(target_t)
        direction_label = "RETROCAUSAL SLIDE (BACKWARDS)" if target_t < 3.0 else "FORWARD SLIDE"
        render_ascii_field(reconstructed_state, f"TIME-SLIDER: {direction_label}", target_t)

    print(f"{BOLD}{'=' * 80}{RESET}")
    print(f"{BOLD} POWDER WORLD & CAUSAL MANIFOLD WEAVING DEMONSTRATION COMPLETE {RESET}")
    print(f"{BOLD}{'=' * 80}{RESET}\n")


if __name__ == "__main__":
    main()
