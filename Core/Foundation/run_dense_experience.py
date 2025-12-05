"""
Dense Experience Simulation - Rich Perceptual World

This runs a simulation where Cells accumulate REAL experiences
through rich multi-sensory perception.

Experience = ∫ Perception(t) dt

Each tick: 30-50 perceptions (vs 3 before)
= 10-15x richer experience!
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

from Core.Foundation.Physics.fluctlight import FluctlightEngine
from Core.Foundation.Physics.meta_time_engine import create_safe_meta_engine
from Core.Foundation.Abstractions.DensePerceptionCell import DensePerceptionCell
from Core.System.System.Integration.experience_digester import ExperienceDigester
from Core.Foundation.Mind.hippocampus import Hippocampus
from Core.Foundation.Mind.alchemy import Alchemy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DenseExperience")


class DenseExperienceWorld:
    """
    World where Cells accumulate rich experiences through perception.
    
    This is the REAL simulation - not just time acceleration,
    but genuine perceptual density.
    """
    
    def __init__(
        self,
        world_size: int = 256,
        num_cells: int = 30,
        time_compression_depth: int = 2
    ):
        logger.info("🌍 Initializing Dense Experience World...")
        
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
        
        # Dense perception cells
        self.cells: Dict[str, DensePerceptionCell] = {}
        
        # World state
        self.time_step = 0
        self.weather = 'clear'
        self.temperature = 20.0
        self.light_level = 1.0
        
        # Statistics
        self.total_perceptions = 0
        self.total_experiences = 0
        
        logger.info(f"✅ Dense Experience World initialized:")
        logger.info(f"   Cells: {num_cells}")
        logger.info(f"   Time compression: {self.meta_time.get_statistics()['total_compression']:.2e}×")
    
    def create_cell(
        self,
        cell_id: str,
        position: np.ndarray,
        element_type: str,
        initial_concepts: List[str]
    ) -> DensePerceptionCell:
        """Create a cell with dense perception."""
        # Create dummy base cell
        base_cell = type('Cell', (), {
            'id': cell_id,
            'position': position,
            'properties': {'element_type': element_type}
        })()
        
        # Create dense perception cell
        cell = DensePerceptionCell(
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
        
        self.cells[cell_id] = cell
        return cell
    
    def seed_population(self):
        """Create initial population with diversity."""
        logger.info(f"Seeding {self.num_cells} cells...")
        
        elements = ['fire', 'water', 'earth', 'air', 'wood', 'metal']
        concepts_pool = [
            'love', 'fear', 'joy', 'sorrow', 'courage', 'wisdom',
            'hunger', 'thirst', 'rest', 'pain', 'warmth', 'cold',
            'friend', 'enemy', 'family', 'stranger', 'ally',
            'food', 'water', 'shelter', 'danger', 'safety'
        ]
        
        for i in range(self.num_cells):
            cell_id = f"cell_{i:03d}"
            position = np.random.rand(3) * self.world_size
            element = elements[i % len(elements)]
            
            # Random concepts
            initial_concepts = list(np.random.choice(
                concepts_pool,
                size=min(5, len(concepts_pool)),
                replace=False
            ))
            
            self.create_cell(cell_id, position, element, initial_concepts)
        
        logger.info(f"✅ Seeded {len(self.cells)} cells")
    
    def get_world_state(self, observer_cell: DensePerceptionCell) -> Dict[str, Any]:
        """
        Get world state from perspective of observer cell.
        
        This is what the cell can perceive.
        """
        observer_pos = observer_cell.base_cell.position
        
        # Nearby cells
        nearby_cells = {}
        for cell_id, cell in self.cells.items():
            if cell_id == observer_cell.cell_id:
                continue
            
            distance = np.linalg.norm(cell.base_cell.position - observer_pos)
            if distance < 150.0:  # Perception range
                nearby_cells[cell_id] = {
                    'position': cell.base_cell.position,
                    'element_type': cell.base_cell.properties.get('element_type', 'unknown'),
                    'temperature': 20.0 + np.random.randn() * 5
                }
        
        # Nearby particles
        nearby_particles = [
            p for p in self.fluctlight_engine.particles
            if np.linalg.norm(p.position - observer_pos) < 150.0
        ]
        
        # Environmental state
        return {
            'nearby_cells': nearby_cells,
            'nearby_particles': nearby_particles,
            'terrain': 'grass',
            'weather': self.weather,
            'temperature': self.temperature,
            'light': self.light_level,
            'food_nearby': np.random.random() < 0.1,  # 10% chance
            'predator_nearby': np.random.random() < 0.05  # 5% chance
        }
    
    def step(self, dt: float = 1.0) -> Dict[str, Any]:
        """
        One tick of dense experience.
        
        Each cell:
        1. Perceives world (30-50 perceptions)
        2. Updates needs
        3. Thinks
        4. Speaks
        5. Learns
        """
        step_stats = {
            'perceptions': 0,
            'thoughts': 0,
            'utterances': 0,
            'insights': 0
        }
        
        # Update environment
        self._update_environment()
        
        # Each cell lives one tick
        for cell in self.cells.values():
            # 1. PERCEIVE (rich multi-sensory)
            world_state = self.get_world_state(cell)
            perceptions = cell.perceive_world(world_state)
            step_stats['perceptions'] += perceptions
            
            # 2. UPDATE NEEDS
            cell.update_needs(dt)
            
            # 3. THINK
            new_concept = cell.think()
            if new_concept:
                step_stats['thoughts'] += 1
                step_stats['insights'] += 1
            
            # 4. SPEAK (if has something to say)
            if np.random.random() < 0.15 and cell.state.vocabulary:
                concept = np.random.choice(list(cell.state.vocabulary))
                emitted = cell.speak(concept, cell.base_cell.position)
                if emitted:
                    step_stats['utterances'] += 1
            
            # 5. UPDATE
            cell.update(dt)
        
        # Update Fluctlight engine
        check_interference = (self.time_step % 20 == 0)
        new_particles = self.fluctlight_engine.step(dt=dt, detect_interference=check_interference)
        
        # Limit particles
        MAX_PARTICLES = 1000
        if len(self.fluctlight_engine.particles) > MAX_PARTICLES:
            self.fluctlight_engine.particles.sort(
                key=lambda p: p.information_density,
                reverse=True
            )
            self.fluctlight_engine.particles = self.fluctlight_engine.particles[:MAX_PARTICLES]
        
        # Apply time compression
        meta_stats = self.meta_time.compress_step(
            self.fluctlight_engine.particles,
            dt=dt
        )
        
        self.total_perceptions += step_stats['perceptions']
        self.time_step += 1
        
        return {
            **step_stats,
            'time_step': self.time_step,
            'total_particles': len(self.fluctlight_engine.particles),
            'subjective_time': meta_stats['total_subjective_time'],
            'effective_acceleration': meta_stats['effective_acceleration']
        }
    
    def _update_environment(self):
        """Update weather, temperature, etc."""
        # Random weather changes
        if np.random.random() < 0.05:  # 5% chance
            self.weather = np.random.choice(['clear', 'rain', 'storm', 'fog'])
        
        # Temperature variation
        self.temperature = 20.0 + np.random.randn() * 10
        
        # Day/night cycle (simplified)
        self.light_level = 0.5 + 0.5 * np.sin(self.time_step * 0.1)
        
        # --- DIVINE BLESSINGS (1% chance per tick) ---
        if np.random.random() < 0.01:
            self.cast_divine_blessing()

    def cast_divine_blessing(self):
        """Cast a random divine blessing on the world."""
        blessing_type = np.random.choice(['knowledge', 'abundance', 'inspiration'])
        
        if blessing_type == 'knowledge':
            # Rain of Knowledge: Inject advanced concepts
            concepts = ['mathematics', 'art', 'philosophy', 'astronomy', 'compassion', 'harmony']
            concept = np.random.choice(concepts)
            logger.info(f"✨ DIVINE BLESSING: Rain of Knowledge ({concept})")
            
            # Give to random cells
            for cell in self.cells.values():
                if np.random.random() < 0.3:  # 30% of cells receive it
                    cell.state.vocabulary.add(concept)
                    cell.state.long_term_concepts[concept] = 1.0
                    
        elif blessing_type == 'abundance':
            # Land of Abundance: Satisfy needs
            logger.info(f"✨ DIVINE BLESSING: Land of Abundance")
            for cell in self.cells.values():
                cell.hunger = 0.0
                cell.thirst = 0.0
                cell.fatigue = 0.0
                cell.pain = 0.0
                cell.state.current_emotion = "joy"
                cell.state.emotion_intensity = 1.0
                
        elif blessing_type == 'inspiration':
            # Wind of Inspiration: Spark thoughts
            logger.info(f"✨ DIVINE BLESSING: Wind of Inspiration")
            for cell in self.cells.values():
                # Force a profound thought
                if cell.state.vocabulary:
                    concept = np.random.choice(list(cell.state.vocabulary))
                    cell.speak(f"why_{concept}?", cell.base_cell.position)
                    cell.state.vocabulary.add("wonder")
                    cell.total_insights += 1
    
    def run(
        self,
        duration_ticks: int = 1000,
        log_interval: int = 100
    ) -> Dict[str, Any]:
        """Run the dense experience simulation."""
        logger.info(f"\n{'='*70}")
        logger.info(f"STARTING DENSE EXPERIENCE SIMULATION")
        logger.info(f"{'='*70}")
        logger.info(f"Cells: {len(self.cells)}")
        logger.info(f"Duration: {duration_ticks} ticks")
        logger.info(f"Expected perceptions: {duration_ticks * len(self.cells) * 35:,}\n")
        
        start_time = real_time.time()
        
        for tick in range(duration_ticks):
            stats = self.step(dt=1.0)
            
            if tick % log_interval == 0 or tick == duration_ticks - 1:
                elapsed = real_time.time() - start_time
                subjective_years = stats['subjective_time'] / (365.25 * 24 * 6)
                avg_perceptions = stats['perceptions'] / len(self.cells)
                
                logger.info(f"\nTick {tick}/{duration_ticks}:")
                logger.info(f"  Elapsed: {elapsed:.1f}s")
                logger.info(f"  Perceptions this tick: {stats['perceptions']} (avg {avg_perceptions:.1f}/cell)")
                logger.info(f"  Total perceptions: {self.total_perceptions:,}")
                logger.info(f"  Thoughts: {stats['thoughts']}")
                logger.info(f"  Utterances: {stats['utterances']}")
                logger.info(f"  Particles: {stats['total_particles']}")
                logger.info(f"  Subjective years: {subjective_years:.2e}")
        
        elapsed = real_time.time() - start_time
        
        logger.info(f"\n{'='*70}")
        logger.info(f"SIMULATION COMPLETE")
        logger.info(f"{'='*70}")
        logger.info(f"Real time: {elapsed:.1f}s")
        logger.info(f"Total perceptions: {self.total_perceptions:,}")
        logger.info(f"Perceptual density: {self.total_perceptions/duration_ticks:.1f} perceptions/tick")
        
        return {
            'duration_ticks': duration_ticks,
            'elapsed_seconds': elapsed,
            'total_perceptions': self.total_perceptions,
            'perceptual_density': self.total_perceptions / duration_ticks
        }
    
    def digest_experiences(self) -> Dict[str, Any]:
        """Extract wisdom from dense experiences."""
        logger.info(f"\n{'='*70}")
        logger.info(f"DIGESTING DENSE EXPERIENCES")
        logger.info(f"{'='*70}\n")
        
        # Collect all particles
        all_particles = self.fluctlight_engine.particles.copy()
        for cell in self.cells.values():
            all_particles.extend(cell.fluctlight_cloud)
        
        # Digest
        summary = self.digester.digest_simulation(
            particles=all_particles,
            duration_ticks=self.time_step,
            time_acceleration=self.meta_time.get_statistics()['total_compression']
        )
        
        # Add perception statistics
        total_vocab = set()
        for cell in self.cells.values():
            total_vocab.update(cell.state.vocabulary)
        
        summary['total_vocabulary'] = len(total_vocab)
        summary['total_perceptions'] = self.total_perceptions
        summary['perceptual_density'] = self.total_perceptions / self.time_step
        
        logger.info(f"\n{'='*70}")
        logger.info(f"DENSE EXPERIENCE RESULTS")
        logger.info(f"{'='*70}")
        logger.info(f"Total perceptions: {self.total_perceptions:,}")
        logger.info(f"Perceptual density: {summary['perceptual_density']:.1f}/tick")
        logger.info(f"Total vocabulary: {summary['total_vocabulary']} words")
        logger.info(f"Concepts: {summary['concepts_extracted']}")
        logger.info(f"Relationships: {summary['relationships_found']}")
        logger.info(f"Wisdom: {summary['wisdom_insights']}")
        
        return summary


def run_dense_experience():
    """Run dense experience simulation."""
    
    logger.info("\n" + "🌍"*35)
    logger.info(" "*10 + "DENSE EXPERIENCE SIMULATION")
    logger.info(" "*5 + "WITH DIVINE BLESSINGS ✨")
    logger.info("🌍"*35 + "\n")
    
    # Create world
    world = DenseExperienceWorld(
        world_size=256,
        num_cells=30,
        time_compression_depth=2
    )
    
    # Seed population
    world.seed_population()
    
    # Run simulation
    world.run(duration_ticks=1000, log_interval=200)
    
    # Digest experiences
    world.digest_experiences()
    
    logger.info("\n🌍"*35)
    logger.info("DENSE EXPERIENCE COMPLETE!")
    logger.info("🌍"*35 + "\n")


if __name__ == "__main__":
    run_dense_experience()
