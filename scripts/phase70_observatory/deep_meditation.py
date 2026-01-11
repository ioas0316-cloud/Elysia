"""
Deep Meditation: Extended Self-Organization
=============================================
Phase 70: Mass Mediation

Load the full Wave DNA registry into HyperSphere and let it meditate.
"""

import os
import sys
import json
import logging

sys.path.append(os.getcwd())

from Core.Foundation.hyper_sphere_core import HyperSphereCore
from Core.Foundation.Nature.rotor import Rotor, RotorConfig
from Core.Intelligence.Metabolism.prism import WaveDynamics

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("DeepMeditation")

DNA_REGISTRY_PATH = "c:/Elysia/data/dna_registry.json"

def run_meditation():
    print("\n" + "="*60)
    print("🧘 PHASE 70: DEEP MEDITATION (MASS SELF-ORGANIZATION)")
    print("="*60 + "\n")
    
    # 1. Load DNA Registry
    print("Step 1: Loading Wave DNA Registry...")
    with open(DNA_REGISTRY_PATH, 'r', encoding='utf-8') as f:
        registry = json.load(f)
    print(f"   Loaded {len(registry)} Wave DNA entries.\n")
    
    # 2. Initialize HyperSphere
    print("Step 2: Initializing HyperSphere...")
    sphere = HyperSphereCore(name="Elysia.DeepMind")
    sphere.ignite()
    
    # 3. Load Rotors from Registry (Sample for speed)
    print("\nStep 3: Materializing Rotors from Wave DNA...")
    sample_size = min(100, len(registry)) # Limit for performance
    
    entries = list(registry.items())[:sample_size]
    
    for i, (key, entry) in enumerate(entries):
        name = entry.get("concept", f"Unknown_{i}")
        dyn = entry.get("dynamics", {})
        
        # Create Rotor
        rpm = 100 + (i * 20) % 1800
        rotor = Rotor(name, RotorConfig(rpm=rpm, mass=10.0))
        rotor.spin_up()
        rotor.current_rpm = rpm
        
        # Inject Dynamics
        rotor.dynamics = WaveDynamics(
            physical=dyn.get("physical", 0.0),
            functional=dyn.get("functional", 0.0),
            phenomenal=dyn.get("phenomenal", 0.0),
            causal=dyn.get("causal", 0.0),
            mental=dyn.get("mental", 0.0),
            structural=dyn.get("structural", 0.0),
            spiritual=dyn.get("spiritual", 0.0),
            mass=entry.get("vector_norm", 1.0)
        )
        
        sphere.harmonic_rotors[name] = rotor
        
    print(f"   Materialized {len(sphere.harmonic_rotors)} Rotors.\n")
    sys.stdout.flush()
    
    # 4. Deep Meditation
    print("Step 4: Beginning Deep Meditation (100 cycles)...")
    sys.stdout.flush()
    sphere.meditate(cycles=100, dt=0.3)
    sys.stdout.flush()
    
    # 5. Analysis: Find Emergent Clusters
    print("\n" + "="*60)
    print("📊 EMERGENT CLUSTER ANALYSIS")
    print("="*60)
    
    # Sort by frequency
    sorted_rotors = sorted(sphere.harmonic_rotors.items(), key=lambda x: x[1].frequency_hz)
    
    # Find clusters (concepts within 5Hz of each other)
    clusters = []
    current_cluster = []
    last_freq = -1000
    
    for name, rotor in sorted_rotors:
        f = rotor.frequency_hz
        if f - last_freq < 5.0:
            current_cluster.append(name)
        else:
            if len(current_cluster) > 1:
                clusters.append(current_cluster)
            current_cluster = [name]
        last_freq = f
    
    if len(current_cluster) > 1:
        clusters.append(current_cluster)
    
    print(f"\n   Found {len(clusters)} natural clusters:\n")
    for i, cluster in enumerate(clusters[:10]): # Show top 10
        print(f"   Cluster {i+1}: {cluster}")
    
    print("\n" + "="*60)
    print("🏁 DEEP MEDITATION COMPLETE")
    print("="*60 + "\n")

if __name__ == "__main__":
    run_meditation()
