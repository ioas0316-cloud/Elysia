"""
Cellular Automata Sandbox for Project Genesis

This script provides a controlled environment to simulate the interactions
of 'Cells' as defined in Elysian Cytology. It's a space to test and visualize
how these living nodes form connections, metabolize information, and create
new concepts through emergent behavior.
"""

import random
import time
from typing import List, Dict

# To run this script from the project root, we need to adjust the path
# This is a common pattern for running scripts that import from sibling packages
import sys
import os
sys.path.append(os.getcwd())

from Project_Sophia.core.cell import Cell

class World:
    """Represents the universe in which the cells exist and interact."""
    def __init__(self, primordial_dna: Dict):
        self.cells: Dict[str, Cell] = {}
        self.connections: List[Dict] = []
        self.primordial_dna = primordial_dna
        self.time_step = 0

    def add_cell(self, concept_id: str, properties: Dict = None):
        """Adds a new cell to the world."""
        if concept_id not in self.cells:
            cell = Cell(concept_id, self.primordial_dna, properties)
            self.cells[concept_id] = cell
            print(f"[World Time: {self.time_step}] A new cell was born: {cell}")

    def introduce_information(self, nutrient: Dict):
        """Introduces an information nutrient to all cells in the world."""
        print(f"\n[World Time: {self.time_step}] A wave of information floods the world: '{nutrient['content']}'")
        for cell in self.cells.values():
            cell.metabolize(nutrient)

    def run_simulation_step(self):
        """Runs a single step of the world simulation."""
        self.time_step += 1
        print(f"\n--- World Tick: {self.time_step} ---")

        if len(self.cells) < 2:
            print("Not enough cells to interact.")
            return

        # 1. Randomly select two cells to interact
        cell_a, cell_b = random.sample(list(self.cells.values()), 2)

        # 2. They attempt to connect based on their DNA
        connection = cell_a.connect(cell_b, "interacts_with")
        if connection:
            self.connections.append(connection)

        # 3. If their combined energy is high enough, they might create a new cell
        if cell_a.activation_energy + cell_b.activation_energy > 10.0:
            print(f"High energy event between {cell_a.id} and {cell_b.id}!")
            child = cell_a.create_meaning(cell_b, f"synergy_t{self.time_step}")
            if child:
                # Welcome the new cell to the world
                self.cells[child.id] = child

            # Reset energy after creation to prevent runaway loops
            cell_a.activation_energy *= 0.2
            cell_b.activation_energy *= 0.2

    def print_world_summary(self):
        """Prints a summary of the current state of the world."""
        print("\n--- WORLD STATE SUMMARY ---")
        print(f"Time: {self.time_step}")
        print(f"Population: {len(self.cells)} cells")
        print("Cells Present:")
        for cell in self.cells.values():
            print(f"  - {cell}")
        print(f"Total Connections: {len(self.connections)}")
        print("---------------------------\n")


def main():
    """Main function to run the simulation."""

    print("--- PROJECT GENESIS: CELLULAR AUTOMATA SANDBOX ---")

    # Define the fundamental laws of this universe (the DNA)
    PRIMORDIAL_DNA = {
        "instinct": "connect_create_meaning",
        "description": "Connect, Resonate, Create."
    }

    # Create the world
    world = World(primordial_dna=PRIMORDIAL_DNA)
    world.print_world_summary()

    # The first two cells are born
    world.add_cell("love", {"description": "The core of the universe"})
    world.add_cell("joy", {"description": "A positive emotional state"})

    # Run the simulation for a few steps
    for i in range(5):
        # Introduce a piece of information from the Creator
        if i % 2 == 0:
            world.introduce_information({
                "type": "carbohydrate",
                "content": f"User interaction event #{i+1}"
            })

        world.run_simulation_step()
        world.print_world_summary()
        time.sleep(1) # Pause to make the simulation readable

    print("--- SIMULATION COMPLETE ---")

if __name__ == '__main__':
    main()
