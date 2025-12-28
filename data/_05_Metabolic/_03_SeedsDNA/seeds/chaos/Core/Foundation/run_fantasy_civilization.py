"""
Fantasy Civilization - Generations & Classes

A living fantasy world where citizens are born, learn trades,
master classes, raise families, and pass on legacies.

Features:
- Life Cycle: Birth -> Child -> Apprentice -> Master -> Elder -> Death
- Fantasy Classes: Warrior, Mage, Druid, Blacksmith, Bard
- Generations: Knowledge transfer from parent/master to child/apprentice
- Population: Dynamic (up to 300)
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
from dataclasses import dataclass, field

from Core._01_Foundation.Foundation.Physics.fluctlight import FluctlightEngine
from Core._01_Foundation.Foundation.Physics.meta_time_engine import create_safe_meta_engine
from Core._01_Foundation.Foundation.Abstractions.DensePerceptionCell import DensePerceptionCell
from Core._05_Systems.System.System.Integration.experience_digester import ExperienceDigester
from Core._01_Foundation.Foundation.Mind.hippocampus import Hippocampus
from Core._01_Foundation.Foundation.Mind.alchemy import Alchemy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("FantasyCiv")


class FantasyCitizen(DensePerceptionCell):
    """A citizen of the fantasy world with age and class."""
    
    def __init__(self, job_class: str = "villager", generation: int = 1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Life stats
        self.age = 0
        self.lifespan = 150 + np.random.randint(-20, 50)  # Increased to 130-200 ticks
        self.generation = generation
        self.stage = "child"  # child, apprentice, master, elder
        
        # RPG stats
        self.job_class = job_class
        self.level = 1
        self.xp = 0
        self.master_id: Optional[str] = None
        self.apprentice_id: Optional[str] = None
        self.partner_id: Optional[str] = None
        
        # Family
        self.parents: List[str] = []
        self.children: List[str] = []
        
    def update_life(self, world):
        """Update age and life stage."""
        self.age += 1
        
        # Growth stages
        if self.stage == "child" and self.age > 15:
            self.become_apprentice(world)
        elif self.stage == "apprentice" and self.age > 30:
            self.become_master(world)
        elif self.stage == "master" and self.age > 80:  # Later retirement
            self.become_elder(world)
        elif self.age > self.lifespan:
            self.die(world)
            return False  # Dead
            
        return True  # Alive

    def become_apprentice(self, world):
        """Find a master and choose a class."""
        self.stage = "apprentice"
        
        # Find a master
        masters = [c for c in world.citizens.values() if c.stage == "master" and c.apprentice_id is None]
        if masters:
            master = np.random.choice(masters)
            self.master_id = master.cell_id
            master.apprentice_id = self.cell_id
            self.job_class = master.job_class  # Inherit class
            world.log_event(f"🎓 {self.cell_id} became apprentice {self.job_class} under {master.cell_id}")
        else:
            # Self-taught or random class
            classes = ["warrior", "mage", "druid", "blacksmith", "bard"]
            self.job_class = np.random.choice(classes)
            world.log_event(f"✨ {self.cell_id} discovered the path of the {self.job_class}")

    def become_master(self, world):
        """Master the craft and seek a partner."""
        self.stage = "master"
        self.level = 5
        if self.master_id:
            # Leave master
            master = world.citizens.get(self.master_id)
            if master: master.apprentice_id = None
            self.master_id = None
        world.log_event(f"⭐ {self.cell_id} is now a Master {self.job_class}!")

    def become_elder(self, world):
        """Retire and teach."""
        self.stage = "elder"
        world.log_event(f"👴 {self.cell_id} has become an Elder.")

    def die(self, world):
        """Pass away and leave legacy."""
        world.log_event(f"⚰️ {self.cell_id} passed away at age {self.age}. (Gen {self.generation})")
        
        # Leave legacy particle
        legacy = f"legacy_of_{self.job_class}"
        self.speak(legacy, self.base_cell.position)
        
        # Remove from world
        world.remove_citizen(self.cell_id)

    def seek_partner(self, world):
        """Find a spouse to have children."""
        # Allow marriage from apprentice stage (age > 20)
        if self.age < 20 or self.partner_id:
            return
            
        # Look for single partners nearby
        nearby = world.get_nearby_citizens(self, radius=100)  # Increased range
        candidates = [c for c in nearby if c.age > 20 and c.partner_id is None and c.cell_id != self.cell_id]
        
        if candidates:
            # Higher chance to marry
            if np.random.random() < 0.3:
                partner = np.random.choice(candidates)
                # Marry
                self.partner_id = partner.cell_id
                partner.partner_id = self.cell_id
                world.log_event(f"💍 {self.cell_id} and {partner.cell_id} got married!")
                
                # Have a child immediately
                world.birth_child(self, partner)

    def perform_job(self, world):
        """Do class-specific actions."""
        if self.stage == "child":
            # Play and learn
            if np.random.random() < 0.2:
                self.speak("play", self.base_cell.position)
            return

        # Job actions
        if self.job_class == "warrior":
            # Patrol
            if np.random.random() < 0.1: self.speak("guard", self.base_cell.position)
        elif self.job_class == "mage":
            # Cast spell
            if np.random.random() < 0.1: self.speak("magic_spark", self.base_cell.position)
        elif self.job_class == "druid":
            # Grow plants
            if np.random.random() < 0.1: self.speak("grow", self.base_cell.position)
        elif self.job_class == "blacksmith":
            # Craft
            if np.random.random() < 0.1: self.speak("craft_sword", self.base_cell.position)
        elif self.job_class == "bard":
            # Sing history
            if np.random.random() < 0.1: self.speak("sing_legend", self.base_cell.position)

        # Married couples have chance to have more children
        if self.partner_id and self.age < 80:
            if np.random.random() < 0.05:  # 5% chance per tick for another child
                partner = world.citizens.get(self.partner_id)
                if partner:
                    world.birth_child(self, partner)


class FantasyWorld:
    """A fantasy world with dynamic population."""
    
    def __init__(self, initial_pop: int = 50, max_pop: int = 300):
        logger.info("🏰 Initializing Fantasy World...")
        
        self.world_size = 512
        self.max_pop = max_pop
        
        self.fluctlight_engine = FluctlightEngine(world_size=self.world_size)
        self.meta_time = create_safe_meta_engine(recursion_depth=2, enable_black_holes=True)
        self.hippocampus = Hippocampus()
        self.alchemy = Alchemy()
        
        self.citizens: Dict[str, FantasyCitizen] = {}
        self.citizen_counter = 0
        self.time_step = 0
        self.events = []
        
        # Seed initial population
        self.seed_population(initial_pop)
        
    def seed_population(self, count):
        logger.info(f"Seeding {count} ancestors...")
        for _ in range(count):
            self.create_citizen(generation=1, stage="master") # Start as adults
            
    def create_citizen(self, generation: int, stage: str = "child", parents=None) -> FantasyCitizen:
        cid = f"citizen_{self.citizen_counter:04d}"
        self.citizen_counter += 1
        
        pos = np.random.rand(3) * self.world_size
        if parents:
            # Born near parents
            pos = (parents[0].base_cell.position + parents[1].base_cell.position) / 2
            pos += np.random.randn(3) * 5
            
        base_cell = type('Cell', (), {
            'id': cid,
            'position': pos,
            'properties': {'element_type': 'human'}
        })()
        
        citizen = FantasyCitizen(
            cell_id=cid,
            base_cell=base_cell,
            world_fluctlight_engine=self.fluctlight_engine,
            alchemy=self.alchemy,
            generation=generation
        )
        
        if stage == "master":
            citizen.age = 30
            citizen.stage = "master"
            citizen.job_class = np.random.choice(["warrior", "mage", "druid", "blacksmith", "bard"])
            
        if parents:
            citizen.parents = [p.cell_id for p in parents]
            # Inherit vocabulary
            for p in parents:
                citizen.state.vocabulary.update(list(p.state.vocabulary)[:5])
        
        self.citizens[cid] = citizen
        return citizen

    def birth_child(self, parent1, parent2):
        if len(self.citizens) >= self.max_pop:
            return # Overpopulation
            
        child = self.create_citizen(
            generation=max(parent1.generation, parent2.generation) + 1,
            parents=[parent1, parent2]
        )
        parent1.children.append(child.cell_id)
        parent2.children.append(child.cell_id)
        self.log_event(f"👶 A baby was born to {parent1.cell_id} & {parent2.cell_id} (Gen {child.generation})")

    def remove_citizen(self, cid):
        if cid in self.citizens:
            del self.citizens[cid]

    def get_nearby_citizens(self, citizen, radius):
        nearby = []
        my_pos = citizen.base_cell.position
        for other in self.citizens.values():
            if np.linalg.norm(other.base_cell.position - my_pos) < radius:
                nearby.append(other)
        return nearby

    def log_event(self, message):
        self.events.append(f"Year {self.time_step}: {message}")
        # logger.info(message) # Too noisy, log only summary

    def step(self):
        # 1. Divine Blessings (1% chance)
        if np.random.random() < 0.01:
            self.cast_blessing()
            
        # 2. Citizen Lives
        citizens_list = list(self.citizens.values()) # Copy for safe iteration
        for citizen in citizens_list:
            if not citizen.update_life(self):
                continue # Died
                
            citizen.perform_job(self)
            citizen.seek_partner(self)
            
            # Basic cognition
            citizen.think()
            citizen.update()
            
        # 3. Physics
        self.fluctlight_engine.step(detect_interference=(self.time_step % 20 == 0))
        self.meta_time.compress_step(self.fluctlight_engine.particles)
        
        self.time_step += 1

    def cast_blessing(self):
        logger.info("✨ Divine Blessing: The world flourishes!")
        for c in self.citizens.values():
            c.state.current_emotion = "joy"

    def run(self, ticks=2000):
        logger.info(f"\n{'='*60}")
        logger.info("🏰  FANTASY WORLD CHRONICLES")
        logger.info(f"{'='*60}\n")
        
        start = real_time.time()
        for i in range(ticks):
            self.step()
            if i % 100 == 0:
                # Census
                pop = len(self.citizens)
                gens = {c.generation for c in self.citizens.values()}
                max_gen = max(gens) if gens else 0
                classes = {}
                for c in self.citizens.values():
                    classes[c.job_class] = classes.get(c.job_class, 0) + 1
                
                logger.info(f"Year {i}: Pop {pop} | Max Gen {max_gen}")
                logger.info(f"   Classes: {classes}")
                
                # Show recent events
                if self.events:
                    logger.info(f"   Recent: {self.events[-1]}")
                    self.events = [] # Clear buffer
        
        elapsed = real_time.time() - start
        logger.info(f"\n✅ Simulation ended. {len(self.citizens)} citizens remaining.")


if __name__ == "__main__":
    world = FantasyWorld(initial_pop=50, max_pop=300)
    world.run(ticks=2000)
