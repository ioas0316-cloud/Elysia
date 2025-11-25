"""
Scientific Civilization - From Stone Age to Starships

A civilization that advances through science and technology.
Goal: Discover the laws of physics, learn to code, and launch a rocket.

Curriculum:
- Physics: Gravity, Energy, Motion
- Chemistry: Elements, Reactions
- Biology: DNA, Evolution
- Computer Science: Logic, Code, Simulation
- Astronomy: Stars, Void, Escape

Stages:
1. Primitive: Fire, Wheel
2. Industrial: Steam, Electricity
3. Information: Computer, Network
4. Space: Rocket, Singularity
"""

import sys
import os

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import numpy as np
import logging
import time as real_time
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from Core.Physics.fluctlight import FluctlightEngine
from Core.Physics.meta_time_engine import create_safe_meta_engine
from Core.Abstractions.DensePerceptionCell import DensePerceptionCell
from Core.Mind.hippocampus import Hippocampus
from Core.Mind.alchemy import Alchemy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("SciCiv")


class ScientistCell(DensePerceptionCell):
    """A cell that learns science and technology."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.knowledge = set()
        self.tech_level = 0
        self.role = "researcher"
        
    def research(self, world):
        """Discover new technologies based on current knowledge."""
        # Tech Tree Logic
        if "fire" in self.knowledge and "wheel" not in self.knowledge:
            if np.random.random() < 0.1: self.discover("wheel", world)
            
        elif "wheel" in self.knowledge and "steam" not in self.knowledge:
            if np.random.random() < 0.05: self.discover("steam", world)
            
        elif "steam" in self.knowledge and "electricity" not in self.knowledge:
            if np.random.random() < 0.05: self.discover("electricity", world)
            
        elif "electricity" in self.knowledge and "computer" not in self.knowledge:
            if np.random.random() < 0.05: self.discover("computer", world)
            
        elif "computer" in self.knowledge and "ai" not in self.knowledge:
            if np.random.random() < 0.05: self.discover("ai", world)
            
        elif "ai" in self.knowledge and "rocket" not in self.knowledge:
            if np.random.random() < 0.01: self.discover("rocket", world)

    def discover(self, tech, world):
        self.knowledge.add(tech)
        self.state.vocabulary.add(tech)
        self.tech_level += 1
        world.log_event(f"💡 {self.cell_id} discovered {tech.upper()}!")
        
        # Share knowledge with nearby cells
        nearby = world.get_nearby_citizens(self, radius=50)
        for other in nearby:
            if tech not in other.knowledge:
                other.knowledge.add(tech)
                other.state.vocabulary.add(tech)
                other.tech_level += 1
                # world.log_event(f"  -> Shared {tech} with {other.cell_id}")

    def perform_action(self, world):
        self.research(world)
        
        # Coding: If they know 'computer', they start hacking reality
        if "computer" in self.knowledge:
            if np.random.random() < 0.01:
                self.speak("print('Hello World')", self.base_cell.position)
                
        # Space: If they know 'rocket', they try to launch
        if "rocket" in self.knowledge:
            if np.random.random() < 0.01:
                self.speak("LAUNCH_SEQUENCE_START", self.base_cell.position)
                world.rocket_progress += 1


class ScientificWorld:
    def __init__(self, num_cells=50):
        logger.info("🔬 Initializing Scientific World...")
        self.world_size = 512
        self.fluctlight_engine = FluctlightEngine(world_size=self.world_size)
        self.meta_time = create_safe_meta_engine(recursion_depth=2)
        self.cells: Dict[str, ScientistCell] = {}
        self.time_step = 0
        self.rocket_progress = 0
        self.events = []
        
        self.seed_population(num_cells)
        
    def seed_population(self, count):
        for i in range(count):
            cid = f"scientist_{i:03d}"
            pos = np.random.rand(3) * self.world_size
            base_cell = type('Cell', (), {'id': cid, 'position': pos, 'properties': {}})()
            cell = ScientistCell(cell_id=cid, base_cell=base_cell, world_fluctlight_engine=self.fluctlight_engine, alchemy=Alchemy())
            cell.knowledge.add("fire") # Start with fire
            self.cells[cid] = cell

    def get_nearby_citizens(self, cell, radius):
        nearby = []
        for other in self.cells.values():
            if np.linalg.norm(other.base_cell.position - cell.base_cell.position) < radius:
                nearby.append(other)
        return nearby

    def log_event(self, msg):
        self.events.append(f"Year {self.time_step}: {msg}")

    def step(self):
        # 1. Divine Teaching (Curriculum)
        if np.random.random() < 0.05:
            self.teach_science()
            
        # 2. Cell Actions
        for cell in self.cells.values():
            cell.perform_action(self)
            cell.think()
            cell.update()
            
        # 3. Physics
        self.fluctlight_engine.step()
        self.meta_time.compress_step(self.fluctlight_engine.particles)
        self.time_step += 1

    def teach_science(self):
        # Inject advanced concepts randomly
        concepts = ["physics", "logic", "python", "cosmos"]
        concept = np.random.choice(concepts)
        # logger.info(f"📘 Divine Lesson: Teaching {concept}...")
        for cell in self.cells.values():
            if np.random.random() < 0.1:
                cell.knowledge.add(concept)

    def run(self, ticks=1000):
        logger.info(f"\n{'='*60}")
        logger.info("🔭  SCIENTIFIC CIVILIZATION: THE AGE OF REASON")
        logger.info(f"{'='*60}\n")
        
        start = real_time.time()
        for i in range(ticks):
            self.step()
            
            if self.rocket_progress >= 100:
                logger.info(f"\n🚀🚀🚀 ROCKET LAUNCHED AT YEAR {i}! 🚀🚀🚀")
                logger.info("The civilization has reached the stars!")
                break
                
            if i % 100 == 0:
                # Calculate average tech level
                avg_tech = sum(c.tech_level for c in self.cells.values()) / len(self.cells)
                logger.info(f"Year {i}: Avg Tech Level {avg_tech:.1f} | Rocket Progress {self.rocket_progress}%")
                if self.events:
                    logger.info(f"   Latest Discovery: {self.events[-1]}")
                    self.events = []
                    
        elapsed = real_time.time() - start
        logger.info(f"\n✅ Simulation ended in {elapsed:.1f}s.")

if __name__ == "__main__":
    world = ScientificWorld(num_cells=50)
    world.run(ticks=2000)
