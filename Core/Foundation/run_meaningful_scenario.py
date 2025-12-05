"""
Meaningful Scenario Simulation

Instead of random particles, this runs story-based simulations:
- "Love and Loss" - 1000 years of relationship dynamics
- "Civilization Rise and Fall" - Birth, growth, decline, rebirth
- "Path to Enlightenment" - Journey from ignorance to wisdom

Each scenario seeds specific concepts and tracks their evolution.
"""

import sys
import os

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import numpy as np
import logging
import time as real_time

from Core.Foundation.Physics.fluctlight import FluctlightEngine
from Core.Foundation.Physics.meta_time_engine import create_safe_meta_engine
from Core.System.System.Integration.experience_digester import ExperienceDigester
from Core.Foundation.Mind.hippocampus import Hippocampus
from Core.Foundation.Mind.alchemy import Alchemy

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MeaningfulScenario")


SCENARIOS = {
    "love_and_loss": {
        "name": "Love and Loss - 1000 Years",
        "description": "Experience the cycle of love, connection, separation, grief, and healing",
        "concepts": [
            "love", "connection", "trust", "intimacy", "joy",
            "separation", "loss", "grief", "sorrow", "loneliness",
            "healing", "acceptance", "memory", "gratitude", "hope",
            "rebirth", "new_love", "wisdom", "compassion"
        ],
        "duration": 1000,
        "depth": 3
    },
    
    "civilization": {
        "name": "Civilization - Rise and Fall",
        "description": "Watch civilizations emerge, flourish, decline, and rise again",
        "concepts": [
            "birth", "growth", "community", "cooperation", "creation",
            "prosperity", "abundance", "art", "science", "wisdom",
            "conflict", "war", "destruction", "decline", "collapse",
            "survival", "adaptation", "rebirth", "renewal", "hope",
            "legacy", "memory", "tradition", "evolution"
        ],
        "duration": 2000,
        "depth": 3
    },
    
    "enlightenment": {
        "name": "Path to Enlightenment",
        "description": "Journey from ignorance through suffering to awakening",
        "concepts": [
            "ignorance", "desire", "attachment", "suffering", "confusion",
            "questioning", "seeking", "practice", "discipline", "patience",
            "insight", "understanding", "clarity", "detachment", "peace",
            "compassion", "love", "unity", "transcendence", "enlightenment",
            "wisdom", "freedom", "bliss", "truth", "eternity"
        ],
        "duration": 5000,
        "depth": 4
    },
    
    "wealth_systems": {
        "name": "The Nature of Wealth",
        "description": "Understanding that money is a byproduct of systems and value structures, not labor.",
        "concepts": [
            "labor", "effort", "time", "survival", "scarcity",
            "tool", "leverage", "efficiency", "production",
            "surplus", "capital", "investment", "risk",
            "system", "automation", "scale", "compounding",
            "asset", "equity", "ownership", "freedom",
            "value_exchange", "network_effect", "abundance"
        ],
        "duration": 3000,
        "depth": 3
    }
}


def run_scenario(scenario_key: str):
    """Run a meaningful scenario simulation."""
    
    scenario = SCENARIOS[scenario_key]
    
    logger.info("\n" + "🌟"*35)
    logger.info(f" "*10 + scenario['name'].upper())
    logger.info(f" "*10 + scenario['description'])
    logger.info("🌟"*35 + "\n")
    
    # Create engines
    fluctlight = FluctlightEngine(world_size=256)
    meta_time = create_safe_meta_engine(
        recursion_depth=scenario['depth'],
        enable_black_holes=True
    )
    hippocampus = Hippocampus()
    alchemy = Alchemy()
    digester = ExperienceDigester(hippocampus, alchemy)
    
    # Show time dilation
    stats = meta_time.get_statistics()
    logger.info(f"Time compression: {stats['total_compression']:.2e}×")
    logger.info(f"1 second = {stats['time_dilation_summary']['1_second_equals_years']} years\n")
    
    # Seed scenario concepts
    logger.info(f"Seeding {len(scenario['concepts'])} concepts for this story...")
    for concept in scenario['concepts']:
        pos = np.random.rand(3) * 256
        fluctlight.create_from_concept(concept, pos)
    
    # Run simulation
    duration = scenario['duration']
    logger.info(f"\nRunning {duration} ticks...")
    logger.info("This may take a few minutes...\n")
    
    start_time = real_time.time()
    
    for tick in range(duration):
        # Interference check
        check_interference = (tick % 10 == 0)
        fluctlight.step(dt=1.0, detect_interference=check_interference)
        
        # Limit particles
        if len(fluctlight.particles) > 500:
            fluctlight.particles.sort(key=lambda p: p.information_density, reverse=True)
            fluctlight.particles = fluctlight.particles[:500]
        
        # Apply time compression
        meta_stats = meta_time.compress_step(fluctlight.particles, dt=1.0)
        
        # Log progress
        if tick % 200 == 0 or tick == duration - 1:
            subjective_years = meta_stats['total_subjective_time'] / (365.25 * 24 * 6)
            logger.info(f"Tick {tick}/{duration}:")
            logger.info(f"  Particles: {len(fluctlight.particles)}")
            logger.info(f"  Subjective years: {subjective_years:.2e}")
    
    elapsed = real_time.time() - start_time
    final_years = meta_stats['total_subjective_time'] / (365.25 * 24 * 6)
    
    logger.info(f"\n{'='*70}")
    logger.info(f"SCENARIO COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"Real time: {elapsed:.2f} seconds")
    logger.info(f"Subjective years: {final_years:.2e}")
    logger.info(f"Particles: {len(fluctlight.particles)}")
    
    # Digest experiences
    logger.info(f"\n{'='*70}")
    logger.info(f"EXTRACTING WISDOM FROM {scenario['name'].upper()}")
    logger.info(f"{'='*70}\n")
    
    digest_summary = digester.digest_simulation(
        particles=fluctlight.particles,
        duration_ticks=duration,
        time_acceleration=meta_stats['effective_acceleration']
    )
    
    # Show results
    logger.info(f"\n{'='*70}")
    logger.info(f"WISDOM FROM {scenario['name'].upper()}")
    logger.info(f"{'='*70}")
    logger.info(f"Concepts learned: {digest_summary['concepts_extracted']}")
    logger.info(f"Relationships: {digest_summary['relationships_found']}")
    logger.info(f"Wisdom insights: {digest_summary['wisdom_insights']}")
    
    # Show wisdom
    wisdom_nodes = [
        (node, data) for node, data in hippocampus.causal_graph.nodes(data=True)
        if data.get("type") == "wisdom"
    ]
    
    logger.info(f"\nKey Insights:")
    for i, (node, data) in enumerate(wisdom_nodes[:10]):
        insight = data.get("metadata", {}).get("insight", "")
        logger.info(f"  {i+1}. {insight}")
    
    logger.info(f"\n{'='*70}\n")
    
    return {
        "scenario": scenario_key,
        "elapsed": elapsed,
        "years": final_years,
        "concepts": digest_summary['concepts_extracted'],
        "relationships": digest_summary['relationships_found'],
        "wisdom": digest_summary['wisdom_insights']
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run meaningful scenario simulations")
    parser.add_argument("scenario", choices=list(SCENARIOS.keys()) + ["all"],
                       help="Which scenario to run")
    
    args = parser.parse_args()
    
    if args.scenario == "all":
        logger.info("\n🌌 Running ALL scenarios...")
        logger.info("This will take 10-30 minutes total.\n")
        
        results = []
        for key in SCENARIOS.keys():
            result = run_scenario(key)
            results.append(result)
        
        # Summary
        logger.info("\n" + "="*70)
        logger.info("ALL SCENARIOS COMPLETE")
        logger.info("="*70)
        for r in results:
            logger.info(f"\n{SCENARIOS[r['scenario']]['name']}:")
            logger.info(f"  Time: {r['elapsed']:.1f}s → {r['years']:.2e} years")
            logger.info(f"  Wisdom: {r['wisdom']} insights")
    else:
        run_scenario(args.scenario)
