"""
Ultra-Dense Simulation - Maximum Meaningful Experience

This is the REAL deal. Not just time acceleration, but DENSE experiences.

Configuration:
- 50,000 ticks (10x normal)
- 2000 max particles (4x normal)
- Interference every 2 ticks (5x more frequent)
- Depth 2 (balanced: not too fast, very dense)

Expected:
- Runtime: ~1 hour
- Subjective time: ~1000 billion years
- Concepts: 1000+
- Relationships: 500+
- Wisdom: 20+

This will be MEANINGFUL.
"""

import sys
import os

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import numpy as np
import logging
import time as real_time
import json
import os

from Core._01_Foundation.Foundation.Physics.fluctlight import FluctlightEngine
from Core._01_Foundation.Foundation.Physics.meta_time_engine import create_safe_meta_engine
from Core._05_Systems.System.System.Integration.experience_digester import ExperienceDigester
from Core._01_Foundation.Foundation.Mind.hippocampus import Hippocampus
from Core._01_Foundation.Foundation.Mind.alchemy import Alchemy

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("UltraDense")


def run_ultra_dense_simulation():
    """Run the most meaningful simulation possible."""
    
    logger.info("\n" + "🔥"*35)
    logger.info(" "*10 + "ULTRA-DENSE SIMULATION")
    logger.info(" "*5 + "Maximum Meaningful Experience")
    logger.info("🔥"*35 + "\n")
    
    # Configuration
    TICKS = 50000
    MAX_PARTICLES = 2000
    INTERFERENCE_INTERVAL = 2  # Very frequent!
    DEPTH = 2  # Balanced
    
    logger.info("Configuration:")
    logger.info(f"  Ticks: {TICKS:,}")
    logger.info(f"  Max particles: {MAX_PARTICLES:,}")
    logger.info(f"  Interference: every {INTERFERENCE_INTERVAL} ticks")
    logger.info(f"  Recursion depth: {DEPTH}")
    logger.info(f"\nExpected runtime: ~1 hour")
    logger.info(f"Expected concepts: 1000+")
    logger.info(f"Expected wisdom: 20+\n")
    
    # Create engines
    logger.info("Initializing engines...")
    fluctlight = FluctlightEngine(world_size=256)
    meta_time = create_safe_meta_engine(
        recursion_depth=DEPTH,
        base_compression=1000.0,
        enable_black_holes=True
    )
    hippocampus = Hippocampus()
    alchemy = Alchemy()
    digester = ExperienceDigester(hippocampus, alchemy)
    
    stats = meta_time.get_statistics()
    logger.info(f"\nTime compression: {stats['total_compression']:.2e}×")
    logger.info(f"1 hour real = {stats['total_compression']*3600/31536000:.2e} years\n")
    
    # Seed rich concept set
    concepts = [
        # Fundamental
        "existence", "void", "light", "darkness", "time", "space", "energy", "matter",
        # Emotions
        "love", "hate", "fear", "courage", "joy", "sorrow", "anger", "peace",
        "hope", "despair", "trust", "betrayal", "compassion", "cruelty",
        # Elements
        "fire", "water", "earth", "air", "metal", "wood", "lightning", "ice",
        # Abstract
        "truth", "lies", "beauty", "ugliness", "wisdom", "ignorance", "freedom", "slavery",
        "justice", "injustice", "order", "chaos", "creation", "destruction",
        # Life
        "birth", "death", "growth", "decay", "health", "sickness", "youth", "age",
        # Social
        "family", "stranger", "friend", "enemy", "community", "isolation", "cooperation", "conflict",
        # Spiritual
        "soul", "body", "mind", "spirit", "transcendence", "attachment", "enlightenment", "ignorance"
    ]
    
    logger.info(f"Seeding {len(concepts)} concepts...")
    for concept in concepts:
        pos = np.random.rand(3) * 256
        fluctlight.create_from_concept(concept, pos)
    
    logger.info(f"✅ Seeded {len(fluctlight.particles)} particles\n")
    
    # Run simulation
    logger.info("="*70)
    logger.info("STARTING ULTRA-DENSE SIMULATION")
    logger.info("="*70)
    logger.info("This will take approximately 1 hour.")
    logger.info("Progress will be logged every 1000 ticks.\n")
    
    start_time = real_time.time()
    last_log_time = start_time
    # Prepare run directory for checkpoints and results
    run_dir = os.path.join("runs", f"ultra_dense_{int(start_time)}")
    os.makedirs(run_dir, exist_ok=True)
    
    for tick in range(TICKS):
        # Frequent interference checks for density
        check_interference = (tick % INTERFERENCE_INTERVAL == 0)
        new_particles = fluctlight.step(dt=1.0, detect_interference=check_interference)
        
        # Higher particle limit
        if len(fluctlight.particles) > MAX_PARTICLES:
            fluctlight.particles.sort(key=lambda p: p.information_density, reverse=True)
            removed = len(fluctlight.particles) - MAX_PARTICLES
            fluctlight.particles = fluctlight.particles[:MAX_PARTICLES]
        
        # Apply time compression
        meta_stats = meta_time.compress_step(fluctlight.particles, dt=1.0)
        
        # Log progress every 1000 ticks
        if tick % 1000 == 0 or tick == TICKS - 1:
            current_time = real_time.time()
            elapsed = current_time - start_time
            tick_time = current_time - last_log_time
            last_log_time = current_time
            
            subjective_years = meta_stats['total_subjective_time'] / (365.25 * 24 * 6)
            
            # Estimate remaining time
            if tick > 0:
                avg_tick_time = elapsed / tick
                remaining_ticks = TICKS - tick
                eta_seconds = avg_tick_time * remaining_ticks
                eta_minutes = eta_seconds / 60
            else:
                eta_minutes = 0
            
            logger.info(f"\nTick {tick:,}/{TICKS:,} ({tick/TICKS*100:.1f}%)")
            logger.info(f"  Elapsed: {elapsed/60:.1f} min | ETA: {eta_minutes:.1f} min")
            logger.info(f"  Particles: {len(fluctlight.particles):,}")
            logger.info(f"  Subjective years: {subjective_years:.2e}")
            logger.info(f"  Last 1000 ticks: {tick_time:.1f}s")

            # --- Checkpoint: save a small snapshot for analysis / resume ---
            try:
                checkpoint = {
                    "tick": tick,
                    "elapsed_seconds": elapsed,
                    "particles_count": len(fluctlight.particles),
                    "meta_stats": meta_stats,
                }

                # summarize top particles (avoid storing complex objects)
                top_particles = []
                try:
                    sorted_particles = sorted(fluctlight.particles, key=lambda p: getattr(p, "information_density", 0), reverse=True)[:20]
                    for p in sorted_particles:
                        top_particles.append({
                            "id": getattr(p, "id", None),
                            "info_density": float(getattr(p, "information_density", 0)),
                            "concept": getattr(p, "concept", None)
                        })
                except Exception:
                    top_particles = []

                checkpoint["top_particles"] = top_particles

                ckpt_path = os.path.join(run_dir, f"checkpoint_{tick}.json")
                with open(ckpt_path, "w", encoding="utf-8") as cf:
                    json.dump(checkpoint, cf, ensure_ascii=False, indent=2)
            except Exception:
                logger.exception("Failed to write checkpoint")
    
    elapsed = real_time.time() - start_time
    final_years = meta_stats['total_subjective_time'] / (365.25 * 24 * 6)
    
    logger.info(f"\n{'='*70}")
    logger.info(f"SIMULATION COMPLETE")
    logger.info(f"{'='*70}")
    logger.info(f"Real time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    logger.info(f"Subjective years: {final_years:.2e}")
    logger.info(f"Final particles: {len(fluctlight.particles):,}")
    logger.info(f"Effective acceleration: {meta_stats['effective_acceleration']:.2e}×")
    
    # Digest experiences
    logger.info(f"\n{'='*70}")
    logger.info(f"DIGESTING ULTRA-DENSE EXPERIENCES")
    logger.info(f"{'='*70}\n")
    
    digest_summary = digester.digest_simulation(
        particles=fluctlight.particles,
        duration_ticks=TICKS,
        time_acceleration=meta_stats['effective_acceleration']
    )
    
    # Results
    logger.info(f"\n{'='*70}")
    logger.info(f"ULTRA-DENSE RESULTS")
    logger.info(f"{'='*70}")
    logger.info(f"Concepts learned: {digest_summary['concepts_extracted']}")
    logger.info(f"Relationships: {digest_summary['relationships_found']}")
    logger.info(f"Emotional patterns: {digest_summary['emotional_patterns']}")
    logger.info(f"Wisdom insights: {digest_summary['wisdom_insights']}")
    logger.info(f"Language patterns: {digest_summary['language_patterns']}")
    
    # Show wisdom
    wisdom_nodes = [
        (node, data) for node, data in hippocampus.causal_graph.nodes(data=True)
        if data.get("type") == "wisdom"
    ]
    
    logger.info(f"\n{'='*70}")
    logger.info(f"WISDOM FROM ULTRA-DENSE EXPERIENCE")
    logger.info(f"{'='*70}")
    for i, (node, data) in enumerate(wisdom_nodes):
        insight = data.get("metadata", {}).get("insight", "")
        logger.info(f"\n{i+1}. {insight}")
    
    logger.info(f"\n{'='*70}")
    logger.info(f"✨ ELYSIA LIVED {final_years:.2e} YEARS ✨")
    logger.info(f"✨ WITH {digest_summary['concepts_extracted']} CONCEPTS ✨")
    logger.info(f"✨ AND {digest_summary['wisdom_insights']} WISDOM INSIGHTS ✨")
    logger.info(f"✨ RESONANCE EVENTS: {digest_summary.get('resonance_events_detected', 0)} ✨")
    logger.info(f"{'='*70}\n")
    
    # Save results in both TXT (human-readable) and JSON (machine-readable)
    with open("ultra_dense_results.txt", "w", encoding="utf-8") as f:
        f.write(f"Ultra-Dense Simulation Results\n")
        f.write(f"="*70 + "\n\n")
        f.write(f"Runtime: {elapsed/60:.1f} minutes\n")
        f.write(f"Subjective years: {final_years:.2e}\n")
        f.write(f"Concepts: {digest_summary['concepts_extracted']}\n")
        f.write(f"Relationships: {digest_summary['relationships_found']}\n")
        f.write(f"Wisdom: {digest_summary['wisdom_insights']}\n")
        f.write(f"Resonance Events: {digest_summary.get('resonance_events_detected', 0)}\n\n")
        f.write(f"Wisdom Insights:\n")
        f.write(f"-"*70 + "\n")
        for i, (node, data) in enumerate(wisdom_nodes):
            insight = data.get("metadata", {}).get("insight", "")
            f.write(f"{i+1}. {insight}\n")
    
    logger.info("Results saved to ultra_dense_results.txt")
    
    # Save machine-readable JSON
    json_results = {
        "metadata": {
            "simulation_type": "ultra_dense",
            "timestamp": real_time.time(),
            "duration_seconds": elapsed,
            "duration_minutes": elapsed / 60,
        },
        "configuration": {
            "ticks": TICKS,
            "max_particles": MAX_PARTICLES,
            "interference_interval": INTERFERENCE_INTERVAL,
            "recursion_depth": DEPTH,
        },
        "results": digest_summary,
        "final_stats": {
            "particles_final": len(fluctlight.particles),
            "subjective_years": final_years,
            "effective_acceleration": meta_stats['effective_acceleration'],
        },
        "wisdom": [
            data.get("metadata", {}).get("insight", "")
            for node, data in wisdom_nodes
        ]
    }
    
    json_output_path = f"runs/ultra_dense_{int(real_time.time())}/results.json"
    os.makedirs(os.path.dirname(json_output_path), exist_ok=True)
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(json_results, f, ensure_ascii=False, indent=2)
    logger.info(f"Machine-readable results saved to {json_output_path}")
    
    return digest_summary


if __name__ == "__main__":
    logger.info("\n🔥 STARTING ULTRA-DENSE SIMULATION 🔥\n")
    logger.info("This is the most meaningful simulation possible.")
    logger.info("Grab a coffee. This will take about an hour.\n")
    
    results = run_ultra_dense_simulation()
    
    logger.info("\n🔥"*35)
    logger.info(" "*15 + "SIMULATION COMPLETE!")
    logger.info("🔥"*35 + "\n")
