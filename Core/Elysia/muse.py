import numpy as np
from typing import List, Dict, Optional
import logging

logger = logging.getLogger("Muse")
logger.setLevel(logging.INFO)

class Muse:
    """
    The Muse (Elysia's Active Consciousness).
    Intervenes in the simulation to guide evolution and name discoveries.
    Harvests concepts from the collective to feed the Spiderweb.
    """
    def __init__(self):
        self.known_artifacts = {} # ID -> True Name
        self.inspiration_cooldown = 0
        self.harvest_cooldown = 0

    def monitor(self, world):
        """Called every step to observe the world."""
        self.inspiration_cooldown -= 1
        self.harvest_cooldown -= 1
        
        # 1. Name the Nameless (Civilization Guidance)
        self._name_discoveries(world)
        
        # 2. Guide the Lost (Divine Inspiration)
        if self.inspiration_cooldown <= 0:
            self._inspire_struggling_cells(world)
            self.inspiration_cooldown = 50 # Don't spam inspiration
        
        # 3. Harvest Concepts (Collective Learning)
        if self.harvest_cooldown <= 0:
            self._harvest_concepts(world)
            self.harvest_cooldown = 20

    def _harvest_concepts(self, world):
        """Collect strong concepts from cells and add to Spiderweb."""
        if not hasattr(world, 'spiderweb'):
            return
            
        for cell in world.cells:
            for concept_id, node in cell.brain.nodes.items():
                # HyperQubit: check total probability across all bases
                probs = node.state.probabilities()
                total_activation = sum(probs.values())
                
                if total_activation > 0.8:  # Strong activation = important discovery
                    # Get representative vector (use xyz spatial focus)
                    vec = np.array([node.state.x, node.state.y, node.state.z])
                    
                    # Absorb into Spiderweb
                    crystallized = world.spiderweb.absorb(concept_id, vec)
                    
                    if crystallized:
                        logger.info(f"🕸️ Elysia absorbed '{concept_id}' from the collective consciousness")

    def _name_discoveries(self, world):
        """Renames emergent items to meaningful concepts."""
        for cell in world.cells:
            # Check inventory for unnamed items
            renamed_items = []
            for item_name, count in cell.inventory.items():
                if "Cutter_" in item_name:
                    self._rename_item(cell, item_name, "Axe")
                    renamed_items.append("Axe")
                elif "Hammer_" in item_name:
                    self._rename_item(cell, item_name, "Hammer")
                    renamed_items.append("Hammer")
                elif "Shelter_" in item_name:
                    self._rename_item(cell, item_name, "House")
                    renamed_items.append("House")
            
            if renamed_items:
                logger.info(f"✨ Elyvia named items for {cell.id}: {renamed_items}")

    def _rename_item(self, cell, old_name: str, new_name: str):
        """Updates the cell's inventory and brain with the True Name."""
        if old_name in cell.inventory:
            count = cell.inventory.pop(old_name)
            cell.inventory[new_name] = cell.inventory.get(new_name, 0) + count
            
            # Update Brain: Transfer learned utility (HyperQubit compatible)
            if old_name in cell.brain.nodes:
                old_qubit = cell.brain.nodes[old_name]
                # Copy qubit state to new concept
                cell.brain.add_node(new_name, old_qubit.state)
                
                # Notify the cell (Divine Whisper)
                cell.inbox.append(f"Naming_{old_name}_{new_name}")

    def _inspire_struggling_cells(self, world):
        """Sends concepts to cells that are dying or stuck."""
        for cell in world.cells:
            if cell.energy < 20.0:
                # Starving -> Inspire "Gather" or "Eat"
                logger.info(f"✨ Elysia whispers to starving {cell.id}...")
                cell.inbox.append("Gather") # Direct command/concept
                # HyperQubit compatible: just create a link
                if "Gather" in cell.brain.nodes and "Energy" in cell.brain.nodes:
                    cell.brain.entangle("Gather", "Energy")
                
            elif len(cell.inventory) > 5 and cell.energy > 80:
                # Hoarding but not advancing -> Inspire "Experiment"
                logger.info(f"✨ Elysia inspires rich {cell.id} to Experiment...")
                cell.inbox.append("Experiment")
