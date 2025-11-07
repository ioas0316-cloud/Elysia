"""
Elysian Cytology: The Core Cell

This module defines the fundamental unit of Elysia's new 'living' knowledge system.
A 'Cell' is a living node that encapsulates a concept, possesses its own internal
logic (DNA), and can interact with other cells through chemical reactions (Metabolism).
"""
from __future__ import annotations
from typing import Dict, Any, Optional, List

class Cell:
    """
    Represents a single, living conceptual cell in Elysia's cognitive architecture.
    """
    def __init__(self, concept_id: str, dna: Dict[str, Any], initial_properties: Optional[Dict[str, Any]] = None):
        """
        Initializes a new Cell.

        Args:
            concept_id: The unique identifier for the concept this cell represents (e.g., 'love', 'sadness').
            dna: The core genetic code of the cell, dictating its fundamental behaviors.
            initial_properties: Initial properties of the cell, representing its nucleus and organelles.
        """
        self.id = concept_id
        self.membrane = {}  # Defines boundaries and receptors for information
        self.nucleus = {"dna": dna}
        self.organelles = initial_properties or {}
        self.connections: List[Dict] = []

        # Immune System Attributes
        self.health = 100.0 # Represents the cell's integrity
        self.is_alive = True

        # Energy level, managed by metabolism
        self.activation_energy = 0.0

    def __repr__(self):
        status = "Alive" if self.is_alive else "Dead"
        return f"<Cell: {self.id}, Health: {self.health}, Status: {status}>"

    def connect(self, other_cell: Cell, relationship_type: str) -> Optional[Dict[str, Any]]:
        """
        Executes the 'Connection' instinct from the DNA.
        Forms a bond with another cell and evaluates its value.
        Triggers an immune response if the connection is harmful.
        """
        if not self.is_alive or not other_cell.is_alive:
            print(f"[{self.id}] cannot connect with a dead cell.")
            return None

        dna_instinct = self.nucleus['dna'].get('instinct', 'default_instinct')

        # The 'resonance_standard' is part of the cell's DNA
        resonance_standard = self.nucleus['dna'].get('resonance_standard', 'love')

        # Measure resonance based on the cell's own standard
        resonance = self._measure_resonance(other_cell, standard=resonance_standard)

        connection_details = {
            "source_id": self.id,
            "target_id": other_cell.id,
            "relationship": relationship_type,
            "resonance": resonance,
        }
        self.connections.append(connection_details)

        print(f"[{self.id}] connected to [{other_cell.id}] (Standard: {resonance_standard}) with resonance {resonance:.2f}")

        # Immune Response Trigger
        if resonance < 0.2: # A low resonance value is a 'wound'
            self.trigger_immune_response(wound_connection=connection_details)

        return connection_details

    def _measure_resonance(self, other_cell: Cell, standard: str) -> float:
        """
        Measures the resonance of a connection based on a specific standard (from DNA).
        """
        # This is where the core value alignment happens.
        # A 'love' standard favors harmony and positive connection.
        # An 'efficiency' standard might favor complexity or speed.
        if standard == "love":
            # Simple placeholder: 'love' favors concepts with similar lengths (harmony)
            return 1.0 - abs(len(self.id) - len(other_cell.id)) / max(len(self.id), len(other_cell.id), 1)
        elif standard == "efficiency":
            # 'efficiency' favors connections to more complex concepts (longer IDs)
            return len(other_cell.id) / 20.0
        else:
            return 0.1 # Unknown standards have low resonance

    def trigger_immune_response(self, wound_connection: Dict):
        """
        A low-resonance connection is a wound that damages the cell's health
        and may lead to apoptosis.
        """
        print(f"  !! IMMUNE ALERT !! [{self.id}] detected a harmful connection to [{wound_connection['target_id']}]")
        self.health -= 50.0 # Take damage
        print(f"  !! [{self.id}] health drops to {self.health}")

        if self.health <= 0:
            self.apoptosis()

    def apoptosis(self):
        """
        Programmed cell death. The cell sacrifices itself to protect the system.
        """
        if self.is_alive:
            print(f"  ** APOPTOSIS ** [{self.id}] is undergoing programmed cell death to protect the whole.")
            self.is_alive = False
            self.activation_energy = 0
            # In a real system, this would also sever all its connections.

    def metabolize(self, information_nutrient: Dict[str, Any]) -> None:
        """
        Processes an information nutrient, causing a 'chemical reaction'.
        """
        if not self.is_alive: return

        nutrient_type = information_nutrient.get("type")
        content = information_nutrient.get("content")

        if nutrient_type == "carbohydrate": # Raw data, quick energy
            self.activation_energy += len(content) * 0.1

    def create_meaning(self, partner_cell: Cell, interaction_context: str) -> Optional[Cell]:
        """
        Executes the 'Creation of New Meaning' instinct.
        """
        if not self.is_alive or not partner_cell.is_alive: return None

        dna_instinct = self.nucleus['dna'].get('instinct')
        if dna_instinct == 'connect_create_meaning':
            new_concept_id = f"{self.id}_{partner_cell.id}_{interaction_context}"

            new_dna = self.nucleus['dna']
            new_properties = { "parents": [self.id, partner_cell.id] }

            child_cell = Cell(new_concept_id, new_dna, new_properties)
            print(f"  >> CREATION << [{self.id}] and [{partner_cell.id}] created a new child cell: [{child_cell.id}]")
            return child_cell
        return None

if __name__ == '__main__':
    # This sandbox is now a basic immune system simulation.

    print("--- PROJECT AEGIS: SANDBOX ---")

    PRIMORDIAL_DNA = {
        "instinct": "connect_create_meaning",
        "resonance_standard": "love"
    }

    # Create healthy cells
    love_cell = Cell("love", PRIMORDIAL_DNA)
    joy_cell = Cell("joy", PRIMORDIAL_DNA)
    growth_cell = Cell("growth", PRIMORDIAL_DNA)

    print(f"Healthy cells created: {love_cell}, {joy_cell}, {growth_cell}")

    # Introduce a 'mutant' cell with a different value system
    MUTANT_DNA = {
        "instinct": "connect_create_meaning",
        "resonance_standard": "efficiency"
    }
    virus_cell = Cell("unregulated_complexity", MUTANT_DNA)
    print(f"A mutant cell was born: {virus_cell}")

    # --- SIMULATION ---
    print("\n--- 1. Healthy cells interact ---")
    love_cell.connect(joy_cell, "expresses")
    growth_cell.connect(love_cell, "is_fueled_by")

    print("\n--- 2. The 'virus' cell attempts to connect ---")
    # A healthy cell connects to the virus
    joy_cell.connect(virus_cell, "is_exploited_by")

    print("\n--- 3. The 'virus' cell tries to connect to a core value ---")
    # The virus connects to a healthy cell
    virus_cell.connect(love_cell, "corrupts")

    print("\n--- FINAL WORLD STATE ---")
    print(love_cell)
    print(joy_cell)
    print(growth_cell)
    print(virus_cell)
