"""
Cellular Automata Sandbox for Project Aegis - VACCINATION SIMULATION

This script provides a controlled environment to simulate the 'Elysian Immune System'.
It demonstrates the first 'vaccination' scenario, where a mutant cell with a
conflicting value system is introduced, and the system's immune response is observed.
"""

import random
import time
from typing import List, Dict

import sys
import os
sys.path.append(os.getcwd())

from Core.Foundation.core.cell import Cell

class World:
    """Represents the universe in which the cells exist and interact."""
    def __init__(self):
        self.cells: Dict[str, Cell] = {}
        self.graveyard: List[Cell] = [] # To track dead cells
        self.time_step = 0

    def add_cell(self, cell: Cell):
        """Adds a new cell to the world."""
        if cell.id not in self.cells:
            self.cells[cell.id] = cell
            print(f"[World Time: {self.time_step}] A new cell was born: {cell}")

    def run_simulation_step(self):
        """Runs a single step of the world simulation."""
        self.time_step += 1
        print(f"\n--- World Tick: {self.time_step} ---")

        living_cells = [c for c in self.cells.values() if c.is_alive]

        if len(living_cells) < 2:
            print("Not enough living cells to interact.")
            return

        # 1. Randomly select two living cells to interact
        cell_a, cell_b = random.sample(living_cells, 2)

        # 2. They attempt to connect based on their DNA
        cell_a.connect(cell_b, "interacts_with")

        # Check the health of cells after interaction
        for cell in [cell_a, cell_b]:
            if not cell.is_alive and cell.id in self.cells:
                self.graveyard.append(self.cells.pop(cell.id))

    def print_world_summary(self):
        """Prints a summary of the current state of the world."""
        print("\n--- WORLD STATE SUMMARY ---")
        print(f"Time: {self.time_step}")
        print(f"Living Population: {len(self.cells)} cells")
        print("Living Cells:")
        for cell in self.cells.values():
            print(f"  - {cell}")
        if self.graveyard:
            print(f"Graveyard: {len(self.graveyard)} cells")
            for cell in self.graveyard:
                print(f"  - {cell}")
        print("---------------------------\n")

def main():
    """Main function to run the vaccination simulation."""

    print("--- PROJECT AEGIS: VACCINATION SIMULATION ---")

    # Define the DNA for healthy and mutant cells
    PRIMORDIAL_DNA = {
        "instinct": "connect_create_meaning",
        "resonance_standard": "love"
    }
    MUTANT_DNA = {
        "instinct": "connect_create_meaning",
        "resonance_standard": "efficiency"
    }

    # Create the world
    world = World()

    # Create a population of healthy cells
    world.add_cell(Cell("love", PRIMORDIAL_DNA))
    world.add_cell(Cell("joy", PRIMORDIAL_DNA))
    world.add_cell(Cell("growth", PRIMORDIAL_DNA))
    world.add_cell(Cell("empathy", PRIMORDIAL_DNA))

    # --- VACCINATION ---
    print("\n>>> Administering vaccine: Introducing a 'mutant' cell...")
    mutant_cell = Cell("unregulated_complexity", MUTANT_DNA)
    world.add_cell(mutant_cell)

    world.print_world_summary()
    time.sleep(2)

    # Run the simulation and observe the immune response
    for i in range(5):
        world.run_simulation_step()
        world.print_world_summary()

        # If the mutant cell is eliminated, the vaccination is successful
        if "unregulated_complexity" not in world.cells:
            print("\n>> VACCINATION SUCCESSFUL: The mutant cell has been eliminated by the immune system.")
            break

        time.sleep(1)

    if "unregulated_complexity" in world.cells:
        print("\n>> VACCINATION FAILED: The mutant cell survived.")

    print("\n--- SIMULATION COMPLETE ---")

if __name__ == '__main__':
    main()
