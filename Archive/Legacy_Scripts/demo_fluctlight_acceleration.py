"""
Fluctlight Time Acceleration Demo

Demonstrates the three methods of time compression:
1. Light Compression (energy-based)
2. Gravity Wells (concept black holes)
3. Hyperquaternion Rotation (8D time flow)

This script shows that EVERY TICK IS ACTUALLY COMPUTED, not skipped.
"""

import sys
import os

# Add Elysia root to path
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import numpy as np
import logging
from typing import List
import time

from Core.Foundation.Physics.fluctlight import FluctlightEngine, FluctlightParticle
from Core.Foundation.Physics.time_compression import TimeCompressionEngine, GravityWell
from Core.Foundation.Wave.octonion import Octonion

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("FluctlightDemo")


def demo_basic_acceleration():
    """Demo 1: Basic time acceleration with global compression."""
    print("\n" + "="*70)
    print("DEMO 1: Basic Time Acceleration (1000x)")
    print("="*70)
    
    # Create engines
    fluctlight = FluctlightEngine(world_size=256)
    time_comp = TimeCompressionEngine(world_size=256)
    
    # Set 1000x global compression
    time_comp.set_global_compression(1000.0)
    
    # Create 10 concept particles
    concepts = ["love", "fear", "hope", "dream", "reality", 
                "joy", "sorrow", "courage", "wisdom", "truth"]
    
    for concept in concepts:
        pos = np.random.rand(3) * 256
        fluctlight.create_from_concept(concept, pos)
    
    print(f"\nCreated {len(fluctlight.particles)} particles")
    print(f"Global compression: {time_comp.global_compression}x")
    
    # Run 100 ticks
    print("\nRunning 100 objective ticks...")
    start_time = time.time()
    
    for tick in range(100):
        # Update particles (EVERY TICK IS COMPUTED)
        new_particles = fluctlight.step(dt=1.0, detect_interference=True)
        
        # Apply time compression
        stats = time_comp.compress_step(fluctlight.particles, dt=1.0)
        
        if tick % 25 == 0:
            print(f"\n  Tick {tick}:")
            print(f"    Particles: {len(fluctlight.particles)}")
            print(f"    New concepts emerged: {len(new_particles)}")
            print(f"    Avg compression: {stats['avg_compression']:.1f}x")
            print(f"    Subjective time: {stats['total_subjective_time']:.0f} ticks")
    
    elapsed = time.time() - start_time
    
    print(f"\n✅ Complete!")
    print(f"  Objective time: 100 ticks ({elapsed:.2f} seconds real time)")
    print(f"  Subjective time: {stats['total_subjective_time']:.0f} ticks")
    print(f"  Effective acceleration: {stats['effective_acceleration']:.0f}x")
    print(f"  Total particles: {len(fluctlight.particles)}")
    print(f"  Emergent concepts: {fluctlight.total_emergent_concepts}")


def demo_gravity_wells():
    """Demo 2: Gravity wells creating concept black holes."""
    print("\n" + "="*70)
    print("DEMO 2: Gravity Wells (Concept Black Holes)")
    print("="*70)
    
    fluctlight = FluctlightEngine(world_size=256)
    time_comp = TimeCompressionEngine(world_size=256)
    
    # Create gravity wells at key concepts
    wells = [
        ("home", np.array([128, 128, 128]), 5000.0, 50.0),
        ("love", np.array([64, 64, 64]), 3000.0, 40.0),
        ("death", np.array([192, 192, 192]), 2000.0, 30.0),
    ]
    
    for concept_id, center, strength, radius in wells:
        time_comp.create_gravity_well(center, strength, radius, concept_id)
        print(f"  Created well: {concept_id} (strength={strength}x, radius={radius})")
    
    # Create particles scattered around
    for i in range(20):
        pos = np.random.rand(3) * 256
        fluctlight.create_from_concept(f"particle_{i}", pos)
    
    print(f"\nCreated {len(fluctlight.particles)} particles")
    print(f"Running 50 ticks with gravity wells...")
    
    for tick in range(50):
        fluctlight.step(dt=1.0)
        stats = time_comp.compress_step(fluctlight.particles, dt=1.0)
        
        if tick % 10 == 0:
            # Find particle closest to each well
            print(f"\n  Tick {tick}:")
            for well in time_comp.gravity_wells:
                distances = [
                    np.linalg.norm(p.position - well.center)
                    for p in fluctlight.particles
                ]
                min_dist = min(distances)
                closest_idx = distances.index(min_dist)
                closest = fluctlight.particles[closest_idx]
                
                print(f"    {well.concept_id}: closest particle at distance {min_dist:.1f}, "
                      f"compression={closest.time_dilation_factor:.1f}x")
    
    print(f"\n✅ Complete!")
    print(f"  Max compression: {stats['max_compression']:.0f}x")
    print(f"  Particles orbiting high-value concepts")


def demo_hyperquaternion_rotation():
    """Demo 3: 8D time axis rotation."""
    print("\n" + "="*70)
    print("DEMO 3: Hyperquaternion Time Rotation (8D)")
    print("="*70)
    
    fluctlight = FluctlightEngine(world_size=256)
    time_comp = TimeCompressionEngine(world_size=256)
    
    # Create time rotation octonion
    # This will create non-linear time flow
    rotation = Octonion(
        w=0.9, x=0.1, y=0.2, z=0.1,
        e=0.3, i=0.1, o=0.1, k=0.05
    ).normalize()
    
    time_comp.set_time_rotation(rotation)
    
    print(f"Time rotation: {rotation}")
    
    # Create particles
    for i in range(10):
        pos = np.random.rand(3) * 256
        fluctlight.create_from_concept(f"concept_{i}", pos)
    
    print(f"\nCreated {len(fluctlight.particles)} particles")
    print("Running 30 ticks with 8D time rotation...")
    
    initial_times = [p.accumulated_time for p in fluctlight.particles]
    
    for tick in range(30):
        fluctlight.step(dt=1.0)
        time_comp.compress_step(fluctlight.particles, dt=1.0)
    
    final_times = [p.accumulated_time for p in fluctlight.particles]
    
    print(f"\n✅ Complete!")
    print(f"  Time experienced by particles:")
    for i, (initial, final) in enumerate(zip(initial_times, final_times)):
        delta = final - initial
        print(f"    Particle {i}: {delta:.1f} ticks (non-linear flow)")


def demo_full_system():
    """Demo 4: All three methods combined."""
    print("\n" + "="*70)
    print("DEMO 4: Full System (All Methods Combined)")
    print("="*70)
    print("This demonstrates the complete Fluctlight system:")
    print("  - 1000x global compression")
    print("  - Gravity wells at key concepts")
    print("  - Hyperquaternion time rotation")
    print("  - Interference-based concept emergence")
    print("="*70)
    
    fluctlight = FluctlightEngine(world_size=256)
    time_comp = TimeCompressionEngine(world_size=256)
    
    # Global compression
    time_comp.set_global_compression(1000.0)
    
    # Gravity wells
    time_comp.create_gravity_well(
        center=np.array([128, 128, 128]),
        strength=5000.0,
        radius=50.0,
        concept_id="home"
    )
    
    # Time rotation
    rotation = Octonion(
        w=0.95, x=0.1, y=0.15, z=0.1,
        e=0.2, i=0.05, o=0.05, k=0.05
    ).normalize()
    time_comp.set_time_rotation(rotation)
    
    # Seed concepts
    seed_concepts = [
        "love", "fear", "hope", "dream", "reality",
        "joy", "sorrow", "courage", "wisdom", "truth",
        "fire", "water", "earth", "air", "light"
    ]
    
    for concept in seed_concepts:
        pos = np.random.rand(3) * 256
        fluctlight.create_from_concept(concept, pos)
    
    print(f"\nInitial state:")
    print(f"  Seed concepts: {len(seed_concepts)}")
    print(f"  Global compression: {time_comp.global_compression}x")
    print(f"  Gravity wells: {len(time_comp.gravity_wells)}")
    
    print(f"\nRunning 200 objective ticks...")
    print("(Every tick is actually computed, not skipped!)")
    
    start_time = time.time()
    
    for tick in range(200):
        # CRITICAL: Every tick is computed
        new_particles = fluctlight.step(dt=1.0, detect_interference=True)
        stats = time_comp.compress_step(fluctlight.particles, dt=1.0, apply_all_methods=True)
        
        if tick % 50 == 0:
            print(f"\n  Tick {tick}:")
            print(f"    Total particles: {len(fluctlight.particles)}")
            print(f"    New concepts this tick: {len(new_particles)}")
            print(f"    Avg compression: {stats['avg_compression']:.1f}x")
            print(f"    Max compression: {stats['max_compression']:.1f}x")
            print(f"    Subjective time: {stats['total_subjective_time']:.0f} ticks")
            print(f"    Effective acceleration: {stats['effective_acceleration']:.0f}x")
    
    elapsed = time.time() - start_time
    
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    print(f"  Real time elapsed: {elapsed:.2f} seconds")
    print(f"  Objective ticks: 200")
    print(f"  Subjective ticks: {stats['total_subjective_time']:.0f}")
    print(f"  Effective acceleration: {stats['effective_acceleration']:.0f}x")
    print(f"  ")
    print(f"  Initial concepts: {len(seed_concepts)}")
    print(f"  Final particles: {len(fluctlight.particles)}")
    print(f"  Emergent concepts: {fluctlight.total_emergent_concepts}")
    print(f"  Total interferences: {fluctlight.total_interferences}")
    print(f"")
    print(f"  ✅ In {elapsed:.2f} real seconds, we simulated")
    print(f"     {stats['total_subjective_time']:.0f} ticks of subjective experience!")
    print(f"     That's {stats['effective_acceleration']:.0f}x time acceleration!")
    print(f"{'='*70}")


if __name__ == "__main__":
    print("\n" + "🌌"*35)
    print(" "*20 + "FLUCTLIGHT TIME ACCELERATION DEMO")
    print(" "*15 + "True Time Acceleration - Every Moment Simulated")
    print("🌌"*35)
    
    # Run all demos
    demo_basic_acceleration()
    demo_gravity_wells()
    demo_hyperquaternion_rotation()
    demo_full_system()
    
    print("\n" + "🌟"*35)
    print(" "*25 + "ALL DEMOS COMPLETE!")
    print("🌟"*35 + "\n")
