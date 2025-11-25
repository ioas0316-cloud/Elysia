"""
Scientific Civilization - Code-Based Life

A simulation where life is defined by Python code.
Cells carry a genome (source code) that determines their behavior.
Evolution occurs by mutating the Abstract Syntax Tree (AST) of this code.
"""

import sys
import os
import time as real_time
import logging
import numpy as np
import traceback
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from Core.Physics.fluctlight import FluctlightEngine, FluctlightParticle
from Core.Evolution.code_mutator import EvolutionaryCoder, SafetySandbox

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CodeLife")

# --- Default Genome ---
DEFAULT_GENOME = """
def update(cell, world):
    # Default behavior: Move randomly, eat, and communicate
    import random
    
    # 1. Listen
    messages = cell.listen()
    for msg in messages:
        # Simple reaction to hearing something
        if "FOOD" in msg:
            cell.turn_left() # Turn towards sound (simplified)
            
    # 2. Speak
    if cell.energy > 80:
        cell.speak(f"FOOD:{cell.position}")
    
    # 3. Move
    action = random.choice(["move_forward", "turn_left", "turn_right", "rest"])
    if action == "move_forward":
        cell.move_forward()
    elif action == "turn_left":
        cell.turn_left()
    elif action == "turn_right":
        cell.turn_right()
        
    # 4. Eat if possible
    cell.eat()
    
    # 5. Split if enough energy
    if cell.energy > 120:
        cell.split()
"""

class GeneticCell:
    """A cell defined by its code."""
    def __init__(self, cell_id: str, genome: str, position: np.ndarray):
        self.id = cell_id
        self.genome = genome
        self.position = position
        self.energy = 50.0
        self.age = 0
        self.direction = np.random.rand(3)
        self.direction /= np.linalg.norm(self.direction)
        self.inbox: List[str] = [] # Incoming messages
        self.outbox: List[str] = [] # Outgoing messages
        
        # Compile genome
        self.executable = self._compile_genome(genome)
        
    def _compile_genome(self, source: str):
        try:
            # Create a safe scope
            scope = {}
            exec(source, scope)
            if "update" not in scope:
                return None
            return scope["update"]
        except Exception as e:
            # logger.warning(f"Genome compilation failed for {self.id}: {e}")
            return None

    def run(self, world):
        """Execute the genome."""
        if not self.executable:
            return

        try:
            # Sandbox execution
            self.executable(self, world)
            self.energy -= 0.5 # High Metabolic cost (Dark Forest)
        except Exception as e:
            # logger.warning(f"Runtime error in {self.id}: {e}")
            self.energy -= 1.0 # Penalty for crashing
            
        # Clear inbox after processing
        self.inbox = []

    # --- Actions available to the genome ---
    def move_forward(self):
        self.position += self.direction * 1.0
        self.energy -= 0.2

    def turn_left(self):
        # Rotate direction vector (simplified)
        theta = np.radians(15)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))
        self.direction = R.dot(self.direction)

    def turn_right(self):
        theta = np.radians(-15)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))
        self.direction = R.dot(self.direction)
        
    def eat(self):
        # Scarcity: Food is rare but valuable
        import random
        if random.random() < 0.05:
            self.energy += 20.0 
        # Else: Starve 

    def split(self):
        # Signal to world to create offspring
        self.energy /= 2.0
        return True # Signal success

    def speak(self, message: str):
        """Broadcast a message to nearby cells."""
        self.outbox.append(str(message)[:50]) # Limit message length
        self.energy -= 0.5 # Speaking costs energy

    def listen(self) -> List[str]:
        """Check inbox for messages."""
        return self.inbox


class CodeWorld:
    def __init__(self, num_cells=50):
        logger.info("🧬 Initializing Code-Based World...")
        self.world_size = 512
        self.cells: List[GeneticCell] = []
        self.coder = EvolutionaryCoder()
        self.time_step = 0
        
        self.seed_population(num_cells)
        
    def seed_population(self, count):
        for i in range(count):
            self.spawn_cell(DEFAULT_GENOME, parent_pos=None)

    def spawn_cell(self, genome: str, parent_pos: Optional[np.ndarray]):
        cid = f"cell_{self.time_step}_{len(self.cells)}"
        if parent_pos is not None:
            pos = parent_pos + (np.random.rand(3) - 0.5) * 5.0
        else:
            pos = np.random.rand(3) * self.world_size
            
        cell = GeneticCell(cid, genome, pos)
        self.cells.append(cell)

    def step(self):
        new_cells = []
        dead_cells = []
        
        # 1. Process Communication
        # O(N^2) naive broadcasting for now
        for sender in self.cells:
            if sender.outbox:
                for msg in sender.outbox:
                    # Broadcast to nearby
                    for receiver in self.cells:
                        if sender == receiver: continue
                        dist = np.linalg.norm(sender.position - receiver.position)
                        if dist < 50.0: # Hearing range
                            receiver.inbox.append(msg)
                sender.outbox = [] # Clear outbox

        # 2. Run Cells
        for cell in self.cells:
            # Run genome
            result = cell.run(self)
            
            # Check split
            if cell.energy > 100: # Threshold in genome might be different, but physical limit is here
                # Mutate genome
                mutated_genome = self.mutate_genome(cell.genome)
                new_cells.append((mutated_genome, cell.position))
                cell.energy /= 2.0
                
            # Check death
            if cell.energy <= 0:
                dead_cells.append(cell)
                
            cell.age += 1
            
        # Apply updates
        for cell in dead_cells:
            self.cells.remove(cell)
            
        for genome, pos in new_cells:
            self.spawn_cell(genome, pos)
            
        self.time_step += 1

    def mutate_genome(self, genome: str) -> str:
        """Apply AST mutation to the genome."""
        # Create a dummy function wrapper to parse
        # (CodeMutator expects a function AST usually, or we can parse module)
        try:
            # We need to wrap it or parse as module. CodeMutator visits nodes.
            # Let's try to use EvolutionaryCoder logic but adapted for string-to-string
            
            # 1. Parse
            tree = self.coder.evolve_function_ast(genome) # We need to expose this or use helper
            
            # Since EvolutionaryCoder.evolve_function takes a callable, let's use CodeMutator directly
            import ast
            from Core.Evolution.code_mutator import CodeMutator
            
            tree = ast.parse(genome)
            mutator = CodeMutator(intensity=0.1)
            new_tree = mutator.visit(tree)
            ast.fix_missing_locations(new_tree)
            new_source = ast.unparse(new_tree)
            
            if mutator.mutations_log:
                logger.info(f"🧬 Mutation: {mutator.mutations_log[0]}")
                
            return new_source
        except Exception:
            return genome # Fallback

    def run(self, ticks=1000):
        logger.info(f"\n{'='*60}")
        logger.info("💻  CODE-BASED LIFE: EVOLUTION OF SYNTAX")
        logger.info(f"{'='*60}\n")
        
        start = real_time.time()
        for i in range(ticks):
            self.step()
            
            if i % 100 == 0:
                avg_energy = np.mean([c.energy for c in self.cells]) if self.cells else 0
                logger.info(f"Year {i}: Population {len(self.cells)} | Avg Energy {avg_energy:.1f}")
                
                if not self.cells:
                    logger.info("💀 Extinction event.")
                    break
                    
        elapsed = real_time.time() - start
        logger.info(f"\n✅ Simulation ended in {elapsed:.1f}s.")
        
        if self.cells:
            best_cell = max(self.cells, key=lambda c: c.energy)
            logger.info(f"\n🏆 BEST CELL (Energy: {best_cell.energy:.1f}) GENOME:")
            logger.info("-" * 40)
            logger.info(best_cell.genome)
            logger.info("-" * 40)

if __name__ == "__main__":
    world = CodeWorld(num_cells=50)
    world.run(ticks=2000)
