import numpy as np
import time as real_time
import logging
from typing import List, Optional, Dict
from Core.Life.genetic_cell import GeneticCell
from Core.Evolution.code_mutator import EvolutionaryCoder, CodeMutator
from Core.Knowledge.library import Library
from Core.Pantheon.deities import Pantheon
from Core.Civilization.material_system import MaterialSystem, ResourceNode
from Core.Elysia.muse import Muse
from Core.Mind.spiderweb import Spiderweb
from Core.Mind.metacortex import MetaCortex
from Core.Elysia.self_modifier import SelfModifier
import ast
import random

logger = logging.getLogger("CodeWorld")

class CodeWorld:
    def __init__(self, num_cells=50, world_size=100.0):
        logger.info("🧬 Initializing Code-Based World...")
        self.world_size = world_size
        self.cells: List[GeneticCell] = []
        self.coder = EvolutionaryCoder()
        self.time_step = 0
        
        # Knowledge Library
        self.library = Library()
        self.artifacts = self.library.get_artifacts()
        
        # The Pantheon (Global Laws)
        self.pantheon = Pantheon()
        
        # Civilization (Materials)
        self.material_system = MaterialSystem()
        self.resources: List[ResourceNode] = []
        self._spawn_resources(count=50)
        
        # The Spiderweb (Collective Consciousness)
        self.spiderweb = Spiderweb(crystallization_threshold=10)
        
        # The Muse (Divine Intervention)
        self.muse = Muse()
        
        # The MetaCortex (Meta-Consciousness)
        self.metacortex = MetaCortex()
        
        # The SelfModifier (Recursive Self-Improvement)
        self.self_modifier = SelfModifier()
        
        # Default Genome (Legacy, mostly unused by Resonance Brain)
        self.default_genome = """
def update(cell, world):
    pass
"""
        self.seed_population(num_cells)
        
    def _spawn_resources(self, count):
        for _ in range(count):
            pos = np.random.rand(3) * self.world_size
            rtype = random.choice(["Wood", "Stone", "Metal"])
            amount = random.randint(5, 20)
            self.resources.append(ResourceNode(rtype, amount, pos))

    def seed_population(self, count):
        for i in range(count):
            self.spawn_cell(self.default_genome, parent_pos=None)

    def spawn_cell(self, genome: str, parent_pos: Optional[np.ndarray], parent_brain=None):
        cid = f"cell_{self.time_step}_{len(self.cells)}"
        if parent_pos is not None:
            pos = parent_pos + (np.random.rand(3) - 0.5) * 5.0
        else:
            pos = np.random.rand(3) * self.world_size
            
        cell = GeneticCell(cid, genome, pos, parent_brain)
        
        # Inject cultural genome (if no parent brain, i.e., fresh spawn)
        if parent_brain is None and hasattr(self, 'spiderweb'):
            cultural_genome = self.spiderweb.get_cultural_genome()
            if cultural_genome:
                logger.info(f"🕸️ Injecting {len(cultural_genome)} universal truths into {cid}")
                for concept_id, vector in cultural_genome.items():
                    cell.brain.add_node(concept_id, vector)
        
        self.cells.append(cell)

    def step(self):
        new_cells = []
        dead_cells = []
        
        # 0. Apply Global Laws (Pantheon)
        self.pantheon.update(self)
        
        # 0.5 Divine Intervention (The Muse)
        self.muse.monitor(self)
        
        # 0.75 Meta-Cognitive Analysis & Self-Modification
        # Analyze every 50 steps, but propose modifications immediately when conditions are met
        if self.time_step % 50 == 0 and hasattr(self, 'metacortex'):
            self.metacortex.analyze(self)
            
            # Immediately evaluate if self-modification is beneficial
            # No need to wait for arbitrary step 100 - trust the MetaCortex's judgment
            if hasattr(self, 'self_modifier'):
                proposal = self.metacortex.propose_self_modification()
                if proposal:
                    self.self_modifier.evaluate_and_apply(proposal, self)
        
        # 1. Process Communication (O(N^2) naive)
        for sender in self.cells:
            if sender.outbox:
                for msg in sender.outbox:
                    for receiver in self.cells:
                        if sender == receiver: continue
                        dist = np.linalg.norm(sender.position - receiver.position)
                        if dist < 50.0:
                            receiver.inbox.append(msg)
                sender.outbox = []

        # 2. Run Cells
        for cell in self.cells:
            cell.run(self)
            
            # Check split
            if cell.energy > 100:
                mutated_genome = self.mutate_genome(cell.genome)
                # Pass brain for inheritance
                new_cells.append((mutated_genome, cell.position, cell.brain))
                cell.energy /= 2.0
                
            # Check death
            if cell.energy <= 0:
                dead_cells.append(cell)
                
            cell.age += 1
            
        # Apply updates
        for cell in dead_cells:
            self.cells.remove(cell)
            
        for genome, pos, parent_brain in new_cells:
            self.spawn_cell(genome, pos, parent_brain)
            
        self.time_step += 1

    def mutate_genome(self, genome: str) -> str:
        """Apply AST mutation to the genome."""
        # Legacy support
        return genome

    def get_neighbors(self, position: np.ndarray, radius: float) -> List['GeneticCell']:
        """Returns cells within a certain radius (Naive O(N))."""
        neighbors = []
        for cell in self.cells:
            if np.linalg.norm(cell.position - position) <= radius:
                neighbors.append(cell)
        return neighbors

    def get_nearby_artifacts(self, position: np.ndarray, radius: float) -> List:
        """Returns knowledge artifacts within radius."""
        found = []
        for artifact in self.artifacts:
            if np.linalg.norm(artifact.position - position) <= radius:
                found.append(artifact)
        return found

    def get_nearby_resources(self, position: np.ndarray, radius: float) -> List[ResourceNode]:
        """Returns resource nodes within radius."""
        found = []
        for res in self.resources:
            if np.linalg.norm(res.position - position) <= radius:
                found.append(res)
        return found

    def get_scent(self, position: np.ndarray) -> np.ndarray:
        """Returns a vector pointing towards the nearest food/energy source."""
        center = np.array([self.world_size/2, self.world_size/2, self.world_size/2])
        direction = center - position
        dist = np.linalg.norm(direction)
        if dist < 1e-6: return np.zeros(3)
        return (direction / dist) * (1000.0 / (dist + 1.0)) 

    def get_environment(self, position: np.ndarray) -> Dict[str, float]:
        """Returns environmental properties (Taste/Touch)."""
        center = np.array([self.world_size/2, self.world_size/2, self.world_size/2])
        dist = np.linalg.norm(position - center)
        energy_density = max(0.0, 1.0 - (dist / self.world_size))
        
        return {
            "energy_density": energy_density,
            "temperature": 0.5, 
        }

    def get_state_json(self) -> Dict:
        """Serializes the world state for visualization."""
        cells_data = []
        for cell in self.cells:
            cells_data.append({
                "id": cell.id,
                "x": float(cell.position[0]),
                "y": float(cell.position[1]),
                "z": float(cell.position[2]),
                "energy": float(cell.energy),
                "age": int(cell.age),
            })
            
        return {
            "time_step": self.time_step,
            "cells": cells_data,
            "world_size": self.world_size
        }
