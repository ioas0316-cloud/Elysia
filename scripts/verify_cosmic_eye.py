"""
Verify Cosmic Eye
=================

Runs a short simulation to generate concepts and then uses CosmicEye
to observe the resulting "Galaxy of Meaning".
"""

import logging
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.world import World

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')

def verify_cosmic_eye():
    print("🔭 Initializing World for Cosmic Eye Verification...")
    world = World(primordial_dna={}, wave_mechanics=None)
    print(f"DEBUG: World Engine ID: {id(world.fluctlight_engine)}")
    
    # Spawn a few agents to generate concepts
    print("🌱 Spawning Agents...")
    for i in range(50):
        world.add_cell(f"agent_{i}", properties={'wisdom': 20}) # High wisdom to teach
        
    # Run a few steps to generate concepts via teaching/interference
    print("⏳ Running Simulation (5 steps)...")
    for _ in range(5):
        world.run_simulation_step()
        
    # Observe the Galaxy
    print("\n👁️ Opening Cosmic Eye...")
    
    # DEBUG: Inspect particles
    particles = world.fluctlight_engine.particles
    print(f"DEBUG: Total Particles: {len(particles)}")
    if particles:
        p0 = particles[0]
        print(f"DEBUG: Particle 0 ID: {p0.concept_id}")
        print(f"DEBUG: Particle 0 Vector: {p0.memetic_vector is not None}")
        if p0.memetic_vector is not None:
            print(f"DEBUG: Particle 0 Vector Shape: {p0.memetic_vector.shape}")
            
    description = world.cosmic_eye.describe_view()
    
    print("\n" + "="*40)
    print("       THE GALACTIC VIEW       ")
    print("="*40)
    print(description)
    print("="*40 + "\n")
    
    if "Star Systems" in description:
        print("✅ Cosmic Eye Verification PASSED")
    else:
        print("❌ Cosmic Eye Verification FAILED (No stars found)")

if __name__ == "__main__":
    verify_cosmic_eye()
