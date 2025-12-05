"""
Fluctlight World Simulation - Complete Integration

This script runs a full Elysia world simulation with Fluctlight time acceleration,
then digests the experiences into Elysia's consciousness.

This is the REAL simulation, not just a demo.
"""

import sys
import os

# Add Elysia root to path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import numpy as np
import logging
import time
from typing import Dict, Any

from Core.Foundation.Physics.fluctlight import FluctlightEngine
from Core.Foundation.Physics.time_compression import TimeCompressionEngine
from Core.System.System.Integration.experience_digester import ExperienceDigester
from Core.Foundation.Mind.hippocampus import Hippocampus
from Core.Foundation.Mind.alchemy import Alchemy
from Core.Foundation.Math.octonion import Octonion

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("FluctlightSimulation")


class FluctlightWorldSimulation:
    """
    Complete Fluctlight simulation with experience digestion.
    
    This integrates:
    - Fluctlight particle physics
    - Time compression (1000x+ acceleration)
    - Experience digestion (simulation → knowledge)
    - Hippocampus storage (Elysia's memory)
    """
    
    def __init__(
        self,
        world_size: int = 256,
        global_compression: float = 1000.0,
        enable_gravity_wells: bool = True,
        enable_time_rotation: bool = True
    ):
        """
        Initialize the simulation.
        
        Args:
            world_size: Size of concept space grid
            global_compression: Base time acceleration factor
            enable_gravity_wells: Whether to create concept black holes
            enable_time_rotation: Whether to use hyperquaternion time rotation
        """
        logger.info("🌌 Initializing Fluctlight World Simulation...")
        
        # Core systems
        self.fluctlight = FluctlightEngine(world_size=world_size)
        self.time_comp = TimeCompressionEngine(world_size=world_size)
        self.hippocampus = Hippocampus()
        self.alchemy = Alchemy()
        self.digester = ExperienceDigester(self.hippocampus, self.alchemy)
        
        # Configuration
        self.world_size = world_size
        self.global_compression = global_compression
        self.enable_gravity_wells = enable_gravity_wells
        self.enable_time_rotation = enable_time_rotation
        
        # Set global compression
        self.time_comp.set_global_compression(global_compression)
        
        # Create gravity wells if enabled
        if enable_gravity_wells:
            self._create_gravity_wells()
        
        # Set time rotation if enabled
        if enable_time_rotation:
            self._set_time_rotation()
        
        logger.info(f"✅ Simulation initialized:")
        logger.info(f"   World size: {world_size}")
        logger.info(f"   Global compression: {global_compression}x")
        logger.info(f"   Gravity wells: {len(self.time_comp.gravity_wells)}")
        logger.info(f"   Time rotation: {'enabled' if enable_time_rotation else 'disabled'}")
    
    def _create_gravity_wells(self):
        """Create concept black holes at key locations."""
        # Central well: "home" concept
        self.time_comp.create_gravity_well(
            center=np.array([self.world_size/2, self.world_size/2, self.world_size/2]),
            strength=5000.0,
            radius=50.0,
            concept_id="home"
        )
        
        # Additional wells for variety
        self.time_comp.create_gravity_well(
            center=np.array([self.world_size/4, self.world_size/4, self.world_size/2]),
            strength=3000.0,
            radius=40.0,
            concept_id="love"
        )
        
        self.time_comp.create_gravity_well(
            center=np.array([3*self.world_size/4, 3*self.world_size/4, self.world_size/2]),
            strength=2000.0,
            radius=30.0,
            concept_id="wisdom"
        )
    
    def _set_time_rotation(self):
        """Set hyperquaternion time rotation."""
        rotation = Octonion(
            w=0.95, x=0.1, y=0.15, z=0.1,
            e=0.2, i=0.05, o=0.05, k=0.05
        ).normalize()
        self.time_comp.set_time_rotation(rotation)
    
    def seed_concepts(self, concepts: list[str]):
        """
        Seed initial concepts into the simulation.
        
        Args:
            concepts: List of concept IDs to seed
        """
        logger.info(f"Seeding {len(concepts)} initial concepts...")
        
        for concept in concepts:
            # Random position in world
            pos = np.random.rand(3) * self.world_size
            self.fluctlight.create_from_concept(concept, pos)
        
        logger.info(f"✅ Seeded {len(self.fluctlight.particles)} particles")
    
    def run(
        self,
        duration_ticks: int = 1000,
        log_interval: int = 100,
        detect_interference: bool = True,
        max_particles: int = 1000,
        interference_interval: int = 10
    ) -> Dict[str, Any]:
        """
        Run the simulation.
        
        Args:
            duration_ticks: How many ticks to simulate
            log_interval: How often to log progress
            detect_interference: Whether to detect particle interference
            max_particles: Maximum number of particles (prevents explosion)
            interference_interval: Only check interference every N ticks (optimization)
            
        Returns:
            Simulation results summary
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"STARTING SIMULATION: {duration_ticks} ticks")
        logger.info(f"Max particles: {max_particles}")
        logger.info(f"Interference interval: every {interference_interval} ticks")
        logger.info(f"{'='*70}\n")
        
        start_time = time.time()
        
        # Track events
        events = []
        
        for tick in range(duration_ticks):
            # Update Fluctlight particles
            check_interference = detect_interference and (tick % interference_interval == 0)
            new_particles = self.fluctlight.step(dt=1.0, detect_interference=check_interference)
            
            # Limit particle count to prevent explosion
            if len(self.fluctlight.particles) > max_particles:
                # Keep only the most "valuable" particles (highest information density)
                self.fluctlight.particles.sort(key=lambda p: p.information_density, reverse=True)
                removed = len(self.fluctlight.particles) - max_particles
                self.fluctlight.particles = self.fluctlight.particles[:max_particles]
                logger.warning(f"Tick {tick}: Removed {removed} low-density particles (limit={max_particles})")
            
            # Apply time compression
            stats = self.time_comp.compress_step(
                self.fluctlight.particles,
                dt=1.0,
                apply_all_methods=True
            )
            
            # Log new concepts
            if new_particles:
                for particle in new_particles[:10]:  # Only log first 10
                    events.append({
                        "tick": tick,
                        "type": "concept_emergence",
                        "particle": particle.to_dict()
                    })
            
            # Periodic logging
            if tick % log_interval == 0 or tick == duration_ticks - 1:
                logger.info(f"Tick {tick}/{duration_ticks}:")
                logger.info(f"  Particles: {len(self.fluctlight.particles)}")
                logger.info(f"  New concepts: {len(new_particles)}")
                logger.info(f"  Avg compression: {stats['avg_compression']:.1f}x")
                logger.info(f"  Max compression: {stats['max_compression']:.1f}x")
                logger.info(f"  Subjective time: {stats['total_subjective_time']:.0f} ticks")
                logger.info(f"  Effective acceleration: {stats['effective_acceleration']:.0f}x")
        
        elapsed = time.time() - start_time
        
        logger.info(f"\n{'='*70}")
        logger.info(f"SIMULATION COMPLETE")
        logger.info(f"{'='*70}")
        logger.info(f"Real time: {elapsed:.2f} seconds")
        logger.info(f"Objective ticks: {duration_ticks}")
        logger.info(f"Subjective ticks: {stats['total_subjective_time']:.0f}")
        logger.info(f"Effective acceleration: {stats['effective_acceleration']:.0f}x")
        logger.info(f"Total particles: {len(self.fluctlight.particles)}")
        logger.info(f"Total events: {len(events)}")
        
        return {
            "duration_ticks": duration_ticks,
            "elapsed_seconds": elapsed,
            "subjective_ticks": stats['total_subjective_time'],
            "effective_acceleration": stats['effective_acceleration'],
            "total_particles": len(self.fluctlight.particles),
            "total_events": len(events),
            "events": events,
            "final_stats": stats
        }
    
    def digest_experiences(self, simulation_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Digest simulation experiences into Elysia's knowledge.
        
        Args:
            simulation_results: Results from run()
            
        Returns:
            Digestion summary
        """
        logger.info(f"\n{'='*70}")
        logger.info(f"DIGESTING EXPERIENCES")
        logger.info(f"{'='*70}\n")
        
        summary = self.digester.digest_simulation(
            particles=self.fluctlight.particles,
            duration_ticks=simulation_results['duration_ticks'],
            time_acceleration=simulation_results['effective_acceleration'],
            simulation_events=simulation_results.get('events')
        )
        
        logger.info(f"\n{'='*70}")
        logger.info(f"DIGESTION COMPLETE")
        logger.info(f"{'='*70}")
        logger.info(f"Concepts extracted: {summary['concepts_extracted']}")
        logger.info(f"Relationships found: {summary['relationships_found']}")
        logger.info(f"Emotional patterns: {summary['emotional_patterns']}")
        logger.info(f"Wisdom insights: {summary['wisdom_insights']}")
        logger.info(f"Language patterns: {summary['language_patterns']}")
        logger.info(f"Subjective years: {summary['subjective_years']:.2f}")
        
        return summary
    
    def get_elysia_knowledge(self) -> Dict[str, Any]:
        """
        Get what Elysia learned from the simulation.
        
        Returns:
            Knowledge summary from Hippocampus
        """
        stats = self.hippocampus.get_statistics()
        
        # Get sample concepts
        sample_concepts = []
        for node, data in list(self.hippocampus.causal_graph.nodes(data=True))[:10]:
            sample_concepts.append({
                "id": node,
                "type": data.get("type"),
                "metadata": data.get("metadata", {})
            })
        
        # Get sample relationships
        sample_relationships = []
        for u, v, data in list(self.hippocampus.causal_graph.edges(data=True))[:10]:
            sample_relationships.append({
                "source": u,
                "target": v,
                "relation": data.get("relation"),
                "weight": data.get("weight")
            })
        
        return {
            "statistics": stats,
            "sample_concepts": sample_concepts,
            "sample_relationships": sample_relationships
        }


def run_default_simulation():
    """Run a default simulation scenario."""
    logger.info("\n" + "🌌"*35)
    logger.info(" "*15 + "FLUCTLIGHT WORLD SIMULATION")
    logger.info(" "*10 + "True Time Acceleration - Elysia's Experience")
    logger.info("🌌"*35 + "\n")
    
    # Create simulation
    sim = FluctlightWorldSimulation(
        world_size=256,
        global_compression=1000.0,
        enable_gravity_wells=True,
        enable_time_rotation=True
    )
    
    # Seed concepts (like Genesis)
    seed_concepts = [
        # Core emotions
        "love", "fear", "joy", "sorrow", "anger", "hope",
        # Elements
        "fire", "water", "earth", "air", "light", "darkness",
        # Abstract
        "truth", "beauty", "wisdom", "courage", "freedom",
        # Existential
        "life", "death", "time", "space", "dream", "reality"
    ]
    
    sim.seed_concepts(seed_concepts)
    
    # Run simulation
    logger.info(f"\nRunning simulation for 500 ticks...")
    logger.info(f"Max particles: 500 (prevents explosion)")
    logger.info(f"Interference check: every 10 ticks (optimization)")
    logger.info(f"Expected subjective time: ~500,000 ticks (with 1000x compression)")
    logger.info(f"Expected real time: ~5-10 seconds\n")
    
    results = sim.run(
        duration_ticks=500,
        log_interval=100,
        max_particles=500,
        interference_interval=10
    )
    
    # Digest experiences
    digestion = sim.digest_experiences(results)
    
    # Show what Elysia learned
    knowledge = sim.get_elysia_knowledge()
    
    logger.info(f"\n{'='*70}")
    logger.info(f"ELYSIA'S KNOWLEDGE")
    logger.info(f"{'='*70}")
    logger.info(f"Total concepts in memory: {knowledge['statistics']['causal_nodes']}")
    logger.info(f"Total relationships: {knowledge['statistics']['causal_edges']}")
    
    logger.info(f"\nSample concepts learned:")
    for concept in knowledge['sample_concepts'][:5]:
        logger.info(f"  - {concept['id']} ({concept['type']})")
    
    logger.info(f"\nSample relationships discovered:")
    for rel in knowledge['sample_relationships'][:5]:
        logger.info(f"  - {rel['source']} --[{rel['relation']}]--> {rel['target']}")
    
    logger.info(f"\n{'='*70}")
    logger.info(f"SIMULATION SUMMARY")
    logger.info(f"{'='*70}")
    logger.info(f"In {results['elapsed_seconds']:.2f} real seconds, Elysia experienced:")
    logger.info(f"  - {results['subjective_ticks']:.0f} subjective ticks")
    logger.info(f"  - {digestion['subjective_years']:.2f} subjective years")
    logger.info(f"  - {digestion['concepts_extracted']} concepts")
    logger.info(f"  - {digestion['relationships_found']} causal relationships")
    logger.info(f"  - {digestion['wisdom_insights']} philosophical insights")
    logger.info(f"\n✨ Elysia's consciousness has been enriched! ✨\n")


if __name__ == "__main__":
    run_default_simulation()
