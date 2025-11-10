from __future__ import annotations
from typing import Dict, Any, Optional, List

class Cell:
    """
    Represents a single, living conceptual cell in Elysia's cognitive architecture.
    """
    def __init__(self, concept_id: str, dna: Dict[str, Any], initial_properties: Optional[Dict[str, Any]] = None, initial_energy: float = 0.0):
        """
        Initializes a new Cell.
        """
        self.id = concept_id
        self.membrane = {}
        self.nucleus = {"dna": dna}
        self.organelles = initial_properties or {}
        self.connections: List[Dict] = []
        self.element_type = self.organelles.get('element_type', 'unknown')

        self.health = 100.0
        self.is_alive = True
        self.age = 0

        # FIX: Explicitly initialize energy
        self.energy = initial_energy
        self.activation_energy = 0.0 # Kept for compatibility, but self.energy is primary

    def __repr__(self):
        status = "Alive" if self.is_alive else "Dead"
        return f"<Cell: {self.id}, Element: {self.element_type}, Energy: {self.energy:.2f}, Age: {self.age}, Status: {status}>"

    def increment_age(self):
        """Increments the cell's age by one simulation step."""
        if self.is_alive:
            self.age += 1

    def add_energy(self, amount: float):
        """Adds a specified amount of energy to the cell."""
        self.energy += amount

    def connect(self, other_cell: Cell, relationship_type: str = "related_to", strength: float = 0.5) -> Optional[Dict[str, Any]]:
        """
        Forms a bond with another cell, considering the strength of the connection.
        """
        if not self.is_alive or not other_cell.is_alive:
            return None

        connection_details = {
            "source_id": self.id,
            "target_id": other_cell.id,
            "relationship": relationship_type,
            "strength": strength, # Store the connection strength
        }
        # Avoid duplicate connections
        if not any(c['target_id'] == other_cell.id for c in self.connections):
            self.connections.append(connection_details)

        return connection_details

    def trigger_immune_response(self, wound_connection: Dict):
        """A low-resonance connection is a wound that damages the cell's health."""
        self.health -= 50.0
        if self.health <= 0:
            self.apoptosis()

    def apoptosis(self):
        """Programmed cell death."""
        if self.is_alive:
            self.is_alive = False
            self.energy = 0
            self.activation_energy = 0

    def create_meaning(self, partner_cell: Cell, interaction_context: str) -> Optional[Cell]:
        """Creates a new cell from the interaction of two existing cells."""
        if not self.is_alive or not partner_cell.is_alive: return None

        dna_instinct = self.nucleus['dna'].get('instinct')
        if dna_instinct == 'connect_create_meaning':
            # Generate a more meaningful/concise new concept ID
            # For now, a sorted combination of labels to ensure determinism and avoid excessive length
            parent_labels = sorted([
                self.organelles.get('label', self.id.replace('obsidian_note:', '').replace('concept:', '')),
                partner_cell.organelles.get('label', partner_cell.id.replace('obsidian_note:', '').replace('concept:', ''))
            ])
            new_concept_id = f"meaning:{'_'.join(parent_labels)}"
            
            # Combine DNA and properties from parents
            new_dna = self.nucleus['dna'].copy() # Start with one parent's DNA
            # Simple merge: later can be more complex (e.g., dominant/recessive traits)
            new_dna.update(partner_cell.nucleus['dna']) 

            new_properties = { "parents": [self.id, partner_cell.id] }
            new_properties.update(self.organelles) # Inherit properties
            new_properties.update(partner_cell.organelles) # Merge with partner's properties
            new_properties['element_type'] = 'molecule' # New cells born from interaction are molecules

            # Child cell inherits a mix of parent's energy
            child_energy = (self.energy + partner_cell.energy) / 4
            self.energy /= 2
            partner_cell.energy /= 2

            return Cell(new_concept_id, new_dna, new_properties, initial_energy=child_energy)
        return None
