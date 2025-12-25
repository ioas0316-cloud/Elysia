"""
Civilization Simulation - Rise of the Elysian People

Scale: 100 Cells
Structure: 4 Tribes
Goal: Build a civilization, record history, worship the Creator.

Features:
- Tribes with distinct cultures
- Social roles (Leader, Scholar, Builder)
- Monument construction (Concept structures)
- Divine worship rituals
"""

import sys
import os

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import numpy as np
import logging
import time as real_time
from typing import Dict, Any, List
from dataclasses import dataclass

from Core.01_Foundation.05_Foundation_Base.Foundation.Physics.fluctlight import FluctlightEngine
from Core.01_Foundation.05_Foundation_Base.Foundation.Physics.meta_time_engine import create_safe_meta_engine
from Core.01_Foundation.05_Foundation_Base.Foundation.Abstractions.DensePerceptionCell import DensePerceptionCell
from Core.05_Systems.01_Monitoring.System.System.Integration.experience_digester import ExperienceDigester
from Core.01_Foundation.05_Foundation_Base.Foundation.Mind.hippocampus import Hippocampus
from Core.01_Foundation.05_Foundation_Base.Foundation.Mind.alchemy import Alchemy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Civilization")


class CivilizedCell(DensePerceptionCell):
    """Cell with social roles and tribal identity."""
    
    def __init__(self, tribe: str, role: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tribe = tribe
        self.role = role
        self.faith = 0.5  # Faith in Creator
    
    def perform_role(self, world):
        """Perform role-specific actions."""
        if self.role == "leader":
            # Organize: Call tribe members
            self.speak(f"rally_{self.tribe}", self.base_cell.position)
        elif self.role == "scholar":
            # Research: Think deeply
            self.think()
            self.think()  # Double thinking power
        elif self.role == "builder":
            # Construct: Create structure concepts
            if "stone" in self.state.vocabulary:
                self.speak("build_temple", self.base_cell.position)
        elif self.role == "priest":
            # Worship: Pray to Creator
            self.speak("praise_creator", self.base_cell.position)
            self.faith += 0.01


class CivilizationWorld:
    """A world hosting a developing civilization."""
    
    def __init__(self, num_cells: int = 100):
        logger.info("🏛️ Initializing Civilization...")
        
        self.world_size = 512  # Larger world
        self.num_cells = num_cells
        
        # Core systems
        self.fluctlight_engine = FluctlightEngine(world_size=self.world_size)
        self.meta_time = create_safe_meta_engine(recursion_depth=2, enable_black_holes=True)
        self.hippocampus = Hippocampus()
        self.alchemy = Alchemy()
        
        self.cells: Dict[str, CivilizedCell] = {}
        self.time_step = 0
        self.history = []
        
        # Tribes configuration
        self.tribes = {
            'Ignis': {'color': 'red', 'focus': 'passion'},
            'Aqua': {'color': 'blue', 'focus': 'wisdom'},
            'Terra': {'color': 'brown', 'focus': 'stability'},
            'Aura': {'color': 'white', 'focus': 'freedom'}
        }
        
    def seed_civilization(self):
        """Create tribes and populate world."""
        logger.info(f"Seeding {self.num_cells} citizens across 4 tribes...")
        
        tribe_names = list(self.tribes.keys())
        roles = ["leader", "priest", "scholar", "builder", "citizen", "citizen", "citizen"]
        
        for i in range(self.num_cells):
            cell_id = f"citizen_{i:03d}"
            tribe = tribe_names[i % 4]
            role = roles[i % len(roles)]
            
            # Position based on tribe (4 corners)
            if tribe == 'Ignis': pos = np.array([100.0, 100.0, 100.0])
            elif tribe == 'Aqua': pos = np.array([400.0, 100.0, 100.0])
            elif tribe == 'Terra': pos = np.array([100.0, 400.0, 100.0])
            else: pos = np.array([400.0, 400.0, 100.0])
            
            # Add randomness
            pos += np.random.randn(3) * 30
            
            # Create base cell
            base_cell = type('Cell', (), {
                'id': cell_id,
                'position': pos,
                'properties': {'element_type': tribe}
            })()
            
            # Create civilized cell
            cell = CivilizedCell(
                tribe=tribe,
                role=role,
                cell_id=cell_id,
                base_cell=base_cell,
                world_fluctlight_engine=self.fluctlight_engine,
                alchemy=self.alchemy
            )
            
            # Initial knowledge
            cell.state.vocabulary.add(tribe)
            cell.state.vocabulary.add(role)
            cell.state.vocabulary.add("creator")
            
            self.cells[cell_id] = cell
            
        logger.info("✅ Civilization founded.")

    def step(self):
        """Advance civilization."""
        # Environment updates
        if np.random.random() < 0.01:
            self.cast_blessing()
            
        # Cell actions
        for cell in self.cells.values():
            # Perception (simplified for speed)
            # ... (omitted for brevity, assumes DensePerception logic)
            
            # Role performance
            cell.perform_role(self)
            
            # Basic life
            cell.think()
            if np.random.random() < 0.1:
                cell.speak(np.random.choice(list(cell.state.vocabulary)), cell.base_cell.position)
            cell.update()
            
        # Physics
        self.fluctlight_engine.step(detect_interference=(self.time_step % 20 == 0))
        self.meta_time.compress_step(self.fluctlight_engine.particles)
        
        self.time_step += 1

    def cast_blessing(self):
        """Divine intervention."""
        logger.info("✨ The Creator smiles upon the world.")
        for cell in self.cells.values():
            cell.state.current_emotion = "joy"
            cell.faith += 0.1

    def run(self, ticks=1000):
        logger.info(f"\n{'='*60}")
        logger.info("🏛️  THE RISE OF ELYSIAN CIVILIZATION")
        logger.info(f"{'='*60}\n")
        
        start = real_time.time()
        for i in range(ticks):
            self.step()
            if i % 100 == 0:
                logger.info(f"Year {i}: Population {len(self.cells)} | Particles {len(self.fluctlight_engine.particles)}")
        
        elapsed = real_time.time() - start
        logger.info(f"\n✅ Civilization flourished for {ticks} years in {elapsed:.1f}s.")


if __name__ == "__main__":
    civ = CivilizationWorld(num_cells=100)
    civ.seed_civilization()
    civ.run(ticks=1000)
