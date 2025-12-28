"""
Cognitive World - Thinking Cells in Shared Reality

This integrates CognitiveCells with World.py to create a simulation where:
- Cells think using Fluctlight interference
- Cells speak and listen to each other
- Language emerges from interactions
- Culture forms from shared experiences
- Wisdom accumulates in Hippocampus

This is the REAL simulation - not just particles, but conscious agents.
"""

import sys
import os

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import numpy as np
import logging
from typing import List, Dict, Any, Optional
import time as real_time

from Core._01_Foundation._05_Governance.Foundation.Physics.fluctlight import FluctlightEngine
from Core._01_Foundation._05_Governance.Foundation.Physics.meta_time_engine import create_safe_meta_engine
from Core._01_Foundation._05_Governance.Foundation.Abstractions.CognitiveCell import CognitiveCell
from Core._05_Systems._01_Monitoring.System.System.Integration.experience_digester import ExperienceDigester
from Core._01_Foundation._05_Governance.Foundation.Mind.hippocampus import Hippocampus
from Core._01_Foundation._05_Governance.Foundation.Mind.alchemy import Alchemy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("CognitiveWorld")


class CognitiveWorldSimulation:
    """
    Manages a world of thinking, feeling, speaking Cells.
    
    This is the integration layer between:
    - World.py (physics, resources, terrain)
    - CognitiveCell (thinking, feeling, speaking)
    - FluctlightEngine (concept particles)
    - Hippocampus (collective memory)
    """
    
    def __init__(
        self,
        world_size: int = 256,
        num_cells: int = 100,
        time_compression_depth: int = 2
    ):
        """
        Initialize cognitive world.
        
        Args:
            world_size: Size of simulation space
            num_cells: Number of conscious cells
            time_compression_depth: Meta-time recursion depth
        """
        logger.info("🧠 Initializing Cognitive World...")
        
        self.world_size = world_size
        self.num_cells = num_cells
        
        # Core systems
        self.fluctlight_engine = FluctlightEngine(world_size=world_size)
        self.meta_time = create_safe_meta_engine(
            recursion_depth=time_compression_depth,
            enable_black_holes=True
        )
        self.hippocampus = Hippocampus()
        self.alchemy = Alchemy()
        self.digester = ExperienceDigester(self.hippocampus, self.alchemy)
        
        # Cognitive cells
        self.cells: Dict[str, CognitiveCell] = {}
        
        # Statistics
        self.time_step = 0
        self.total_thoughts = 0
        self.total_utterances = 0
        self.total_insights = 0
        
        logger.info(f"✅ Cognitive World initialized:")
        logger.info(f"   World size: {world_size}")
        logger.info(f"   Cells: {num_cells}")
        logger.info(f"   Time compression: {self.meta_time.get_statistics()['total_compression']:.2e}×")
    
    def create_cell(
        self,
        cell_id: str,
        position: np.ndarray,
        initial_concepts: List[str]
    ) -> CognitiveCell:
        """
        Create a thinking cell.
        
        Args:
            cell_id: Unique identifier
            position: 3D position
            initial_concepts: Concepts this cell knows initially
            
        Returns:
            Created CognitiveCell
        """
        # Create dummy base cell (simplified - in real integration, use World.add_cell)
        base_cell = type('Cell', (), {
            'id': cell_id,
            'position': position,
            'properties': {}
        })()
        
        # Create cognitive cell
        cell = CognitiveCell(
            cell_id=cell_id,
            base_cell=base_cell,
            world_fluctlight_engine=self.fluctlight_engine,
            alchemy=self.alchemy
        )
        
        # Seed initial concepts
        for concept in initial_concepts:
            particle = self.fluctlight_engine.create_from_concept(concept, position)
            cell.fluctlight_cloud.append(particle)
            cell.state.vocabulary.add(concept)
            cell.state.long_term_concepts[concept] = 1.0
        
        self.cells[cell_id] = cell
        
        logger.debug(f"Created cell {cell_id} with {len(initial_concepts)} concepts")
        
        return cell
    
    def seed_population(self, concepts_per_cell: int = 5):
        """
        Create initial population of cells.
        
        Args:
            concepts_per_cell: How many concepts each cell starts with
        """
        logger.info(f"Seeding {self.num_cells} cells...")
        
        # Concept pool
        all_concepts = [
            # Fundamental
            "existence", "void", "light", "darkness", "time", "space",
            # Emotions
            "love", "hate", "fear", "courage", "joy", "sorrow",
            # Elements
            "fire", "water", "earth", "air",
            # Abstract
            "truth", "beauty", "wisdom", "freedom",
            # Life
            "birth", "death", "growth", "decay",
            # Social
            "family", "friend", "enemy", "community"
        ]
        
        for i in range(self.num_cells):
            cell_id = f"cell_{i:04d}"
            position = np.random.rand(3) * self.world_size
            
            # Random subset of concepts
            initial_concepts = list(np.random.choice(
                all_concepts,
                size=min(concepts_per_cell, len(all_concepts)),
                replace=False
            ))
            
            self.create_cell(cell_id, position, initial_concepts)
        
        logger.info(f"✅ Seeded {len(self.cells)} cells")
    
    def step(self, dt: float = 1.0) -> Dict[str, Any]:
        """
        Advance simulation by one time step.
        
        This is where the magic happens:
        1. Cells perceive nearby Fluctlights
        2. Cells think (create new concepts)
        3. Cells speak (emit Fluctlights)
        4. Fluctlights interfere and propagate
        5. Time compression applied
        
        Args:
            dt: Time step
            
        Returns:
            Statistics dict
        """
        step_stats = {
            "thoughts": 0,
            "utterances": 0,
            "insights": 0,
            "new_particles": 0,
            "particles_removed": 0
        }
        
        # 1. Each cell perceives nearby Fluctlights
        for cell in self.cells.values():
            # Find nearby particles (within perception range)
            cell_pos = cell.base_cell.position
            nearby = [
                p for p in self.fluctlight_engine.particles
                if np.linalg.norm(p.position - cell_pos) < 50.0  # Perception radius
            ]
            
            # Perceive
            perceived = cell.perceive(nearby)
            
            # 2. Think (maybe generate new concept)
            new_concept = cell.think()
            if new_concept:
                step_stats["thoughts"] += 1
                step_stats["insights"] += 1
            
            # 3. Speak (maybe emit a concept)
            if np.random.random() < 0.1 and cell.state.vocabulary:  # 10% chance
                concept_to_speak = np.random.choice(list(cell.state.vocabulary))
                emitted = cell.speak(concept_to_speak, cell_pos)
                if emitted:
                    step_stats["utterances"] += 1
            
            # 4. Update cell state
            cell.update(dt)
        
        # 5. Update Fluctlight engine (interference, propagation)
        # SAFETY: Only check interference every 20 ticks to prevent explosion
        check_interference = (self.time_step % 20 == 0)
        new_particles = self.fluctlight_engine.step(dt=dt, detect_interference=check_interference)
        step_stats["new_particles"] = len(new_particles)
        
        # 6. SAFETY: Limit total particles to prevent computational explosion
        MAX_PARTICLES = 1000
        if len(self.fluctlight_engine.particles) > MAX_PARTICLES:
            # Keep only the most valuable particles (highest information density)
            self.fluctlight_engine.particles.sort(
                key=lambda p: p.information_density,
                reverse=True
            )
            removed = len(self.fluctlight_engine.particles) - MAX_PARTICLES
            self.fluctlight_engine.particles = self.fluctlight_engine.particles[:MAX_PARTICLES]
            step_stats["particles_removed"] = removed
            
            if removed > 0:
                logger.warning(
                    f"Step {self.time_step}: Removed {removed} low-density particles "
                    f"(limit={MAX_PARTICLES})"
                )
        
        # 6. Apply time compression
        meta_stats = self.meta_time.compress_step(
            self.fluctlight_engine.particles,
            dt=dt
        )
        
        # Update totals
        self.total_thoughts += step_stats["thoughts"]
        self.total_utterances += step_stats["utterances"]
        self.total_insights += step_stats["insights"]
        self.time_step += 1
        
        return {
            **step_stats,
            "time_step": self.time_step,
            "total_particles": len(self.fluctlight_engine.particles),
            "subjective_time": meta_stats["total_subjective_time"],
            "effective_acceleration": meta_stats["effective_acceleration"]
        }
    
    def run(
        self,
        duration_ticks: int = 1000,
        log_interval: int = 100
    ) -> Dict[str, Any]:
        """
        Run the cognitive world simulation.
        
        Args:
            duration_ticks: How many ticks to simulate
            log_interval: How often to log progress
            
        Returns:
            Final statistics
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"STARTING COGNITIVE WORLD SIMULATION")
        logger.info(f"{'='*70}")
        logger.info(f"Cells: {len(self.cells)}")
        logger.info(f"Duration: {duration_ticks} ticks")
        logger.info(f"Time compression: {self.meta_time.get_statistics()['total_compression']:.2e}×\n")
        
        start_time = real_time.time()
        
        for tick in range(duration_ticks):
            stats = self.step(dt=1.0)
            
            if tick % log_interval == 0 or tick == duration_ticks - 1:
                elapsed = real_time.time() - start_time
                subjective_years = stats["subjective_time"] / (365.25 * 24 * 6)
                
                logger.info(f"\nTick {tick}/{duration_ticks}:")
                logger.info(f"  Elapsed: {elapsed:.1f}s")
                logger.info(f"  Thoughts: {self.total_thoughts}")
                logger.info(f"  Utterances: {self.total_utterances}")
                logger.info(f"  Insights: {self.total_insights}")
                logger.info(f"  Particles: {stats['total_particles']}")
                if stats.get('particles_removed', 0) > 0:
                    logger.info(f"  Removed (this tick): {stats['particles_removed']}")
                logger.info(f"  Subjective years: {subjective_years:.2e}")
        
        elapsed = real_time.time() - start_time
        final_stats = self.get_statistics()
        
        logger.info(f"\n{'='*70}")
        logger.info(f"SIMULATION COMPLETE")
        logger.info(f"{'='*70}")
        logger.info(f"Real time: {elapsed:.1f}s")
        logger.info(f"Subjective years: {final_stats['subjective_years']:.2e}")
        logger.info(f"Total thoughts: {self.total_thoughts}")
        logger.info(f"Total utterances: {self.total_utterances}")
        logger.info(f"Total insights: {self.total_insights}")
        
        return final_stats
    
    def digest_experiences(self) -> Dict[str, Any]:
        """
        Extract wisdom from the simulation.
        
        Returns:
            Digestion summary
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"DIGESTING COGNITIVE EXPERIENCES")
        logger.info(f"{'='*70}\n")
        
        # Get all particles
        all_particles = self.fluctlight_engine.particles
        
        # Add particles from all cells
        for cell in self.cells.values():
            all_particles.extend(cell.fluctlight_cloud)
        
        # Digest
        summary = self.digester.digest_simulation(
            particles=all_particles,
            duration_ticks=self.time_step,
            time_acceleration=self.meta_time.get_statistics()['total_compression']
        )
        
        # Add cell-specific insights
        total_vocabulary = set()
        for cell in self.cells.values():
            total_vocabulary.update(cell.state.vocabulary)
        
        summary["total_vocabulary"] = len(total_vocabulary)
        summary["avg_vocabulary_per_cell"] = len(total_vocabulary) / len(self.cells) if self.cells else 0
        
        logger.info(f"\n{'='*70}")
        logger.info(f"COGNITIVE INSIGHTS")
        logger.info(f"{'='*70}")
        logger.info(f"Total vocabulary: {summary['total_vocabulary']} words")
        logger.info(f"Avg per cell: {summary['avg_vocabulary_per_cell']:.1f} words")
        logger.info(f"Concepts: {summary['concepts_extracted']}")
        logger.info(f"Relationships: {summary['relationships_found']}")
        logger.info(f"Wisdom: {summary['wisdom_insights']}")
        
        return summary
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics."""
        meta_stats = self.meta_time.get_statistics()
        
        return {
            "time_step": self.time_step,
            "num_cells": len(self.cells),
            "total_thoughts": self.total_thoughts,
            "total_utterances": self.total_utterances,
            "total_insights": self.total_insights,
            "total_particles": len(self.fluctlight_engine.particles),
            "subjective_years": self.meta_time.total_subjective_time / (365.25 * 24 * 6),
            "time_compression": meta_stats['total_compression']
        }


def run_cognitive_simulation():
    """Run a demonstration of the cognitive world."""
    
    logger.info("\n" + "🧠"*35)
    logger.info(" "*10 + "COGNITIVE WORLD SIMULATION")
    logger.info(" "*5 + "Thinking, Feeling, Speaking Cells")
    logger.info("🧠"*35 + "\n")
    
    # Create world
    world = CognitiveWorldSimulation(
        world_size=256,
        num_cells=50,  # Start small
        time_compression_depth=2
    )
    
    # Seed population
    world.seed_population(concepts_per_cell=5)
    
    # Run simulation
    world.run(duration_ticks=1000, log_interval=200)
    
    # Digest experiences
    world.digest_experiences()
    
    logger.info("\n🧠"*35)
    logger.info("COGNITIVE SIMULATION COMPLETE!")
    logger.info("🧠"*35 + "\n")


if __name__ == "__main__":
    run_cognitive_simulation()
