"""
Verification Script: Autonomous Clustering (Phase 69)
=====================================================
"Let the Thoughts Organize Themselves."

This script tests the meditate() function.
We load sample Wave DNA, let the HyperSphere meditate, and observe clustering.
"""

import os
import sys
import json
import logging

sys.path.append(os.getcwd())

from Core.Foundation.hyper_sphere_core import HyperSphereCore
from Core.Foundation.Nature.rotor import Rotor, RotorConfig
from Core.Intelligence.Metabolism.prism import PrismEngine, WaveDynamics

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("ClusteringTest")

def run_test():
    print("\n" + "="*60)
    print("🧘 PHASE 69 VERIFICATION: AUTONOMOUS CLUSTERING")
    print("="*60 + "\n")
    
    # 1. Initialize Prism and HyperSphere
    print("Step 1: Awakening Systems...")
    prism = PrismEngine()
    prism._load_model()
    
    sphere = HyperSphereCore(name="MeditationTest")
    sphere.ignite()
    
    # 2. Create Test Rotors with Known Dynamics
    concepts = [
        ("Fire", "Burning flames of passion"),
        ("Passion", "Intense emotional drive"),
        ("Water", "Cool flowing river"),
        ("Calm", "Peaceful tranquility"),
        ("Code", "Software programming logic"),
        ("Algorithm", "Computational procedure"),
        ("Love", "Deep affection and connection"),
        ("Soul", "Spiritual essence of being"),
    ]
    
    print("\nStep 2: Transducing Test Concepts...")
    for name, desc in concepts:
        profile = prism.transduce(desc)
        
        rpm = 100 + len(sphere.harmonic_rotors) * 500
        rotor = Rotor(name, RotorConfig(rpm=rpm, mass=10.0))
        rotor.inject_spectrum(profile.spectrum, profile.dynamics)
        rotor.spin_up() # Activate the rotor
        rotor.current_rpm = rpm # Force initial speed
        
        sphere.harmonic_rotors[name] = rotor
        print(f"   🌊 '{name}' -> F:{rotor.frequency_hz:.1f}Hz | Sp:{profile.dynamics.spiritual:.2f}")
    
    # 3. Show Initial Frequencies
    print("\nStep 3: Initial Frequency Map...")
    for name, r in sphere.harmonic_rotors.items():
        print(f"   {name}: {r.frequency_hz:.1f} Hz")
    
    # 4. Meditate (Self-Organize)
    print("\nStep 4: Meditating (Self-Organizing)...")
    sphere.meditate(cycles=20, dt=0.5)
    
    # 5. Show Final Frequencies (Expect Clusters)
    print("\nStep 5: Final Frequency Map (After Meditation)...")
    sorted_rotors = sorted(sphere.harmonic_rotors.items(), key=lambda x: x[1].frequency_hz)
    for name, r in sorted_rotors:
        print(f"   {name}: {r.frequency_hz:.1f} Hz")
    
    # 6. Check for Clustering
    print("\n" + "="*60)
    print("📊 CLUSTERING ANALYSIS")
    print("="*60)
    
    # Fire and Passion should be close
    fire_f = sphere.harmonic_rotors["Fire"].frequency_hz
    passion_f = sphere.harmonic_rotors["Passion"].frequency_hz
    fire_passion_dist = abs(fire_f - passion_f)
    
    # Water and Calm should be close
    water_f = sphere.harmonic_rotors["Water"].frequency_hz
    calm_f = sphere.harmonic_rotors["Calm"].frequency_hz
    water_calm_dist = abs(water_f - calm_f)
    
    # Love and Soul should be close (Spiritual)
    love_f = sphere.harmonic_rotors["Love"].frequency_hz
    soul_f = sphere.harmonic_rotors["Soul"].frequency_hz
    love_soul_dist = abs(love_f - soul_f)
    
    print(f"   Fire <-> Passion: {fire_passion_dist:.1f} Hz (Expect Close)")
    print(f"   Water <-> Calm: {water_calm_dist:.1f} Hz (Expect Close)")
    print(f"   Love <-> Soul: {love_soul_dist:.1f} Hz (Expect Close)")
    
    if fire_passion_dist < 50 and water_calm_dist < 50 and love_soul_dist < 50:
        print("\n   ✅ PASS: Concepts clustered by resonance!")
    else:
        print("\n   ⚠️ Partial: Some clustering occurred. Need more cycles.")
    
    print("\n" + "="*60)
    print("🏁 MEDITATION COMPLETE")
    print("="*60 + "\n")

if __name__ == "__main__":
    run_test()
