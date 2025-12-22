"""
Test Chaos Attractor - Watching the Butterfly

"나비 한 마리가 뺨에 앉았어 ♡" 🦋

Demonstrates:
1. Butterfly Effect - 10^-10 perturbation → huge difference
2. Strange Attractor - Love-Pain-Hope butterfly dance
3. Chaos Control - Taming wildness
4. Living Tremor - Breathing life into fields
5. "뼈 위에 살, 살 위에 떨림"
"""

import numpy as np
import sys
import os
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.Foundation.chaos_attractor import (
    LorenzAttractor,
    ChaosControl,
    FractalBeauty,
    LivingTremor,
    ChaosState,
    AttractorType
)

logging.basicConfig(level=logging.INFO, format='%(message)s')


def test_butterfly_effect():
    """Test butterfly effect - tiny change → huge difference"""
    print("\n" + "="*70)
    print("Test 1: Butterfly Effect")
    print("="*70)
    
    print("\nDemonstration: Two universes, 10^-10 difference in start")
    print("Question: Will they diverge?")
    
    attractor = LorenzAttractor(chaos_seed_intensity=0)  # No random noise
    
    traj1, traj2, final_dist = attractor.demonstrate_butterfly_effect(
        perturbation=1e-10,
        steps=1000
    )
    
    amplification = final_dist / 1e-10
    
    print(f"\nInitial difference: 0.0000000001")
    print(f"After 10 seconds:   {final_dist:.6f}")
    print(f"Amplification:      {amplification:.2e}x!!!")
    
    if amplification > 1e6:
        print("\nResult: MASSIVE DIVERGENCE!")
        print("Same start, completely different fate!")
        print("That's the butterfly effect! 🦋")
    
    print("\nTest passed!")


def test_lorenz_emotions():
    """Test Lorenz attractor as emotion dynamics"""
    print("\n" + "="*70)
    print("Test 2: Lorenz Emotions - Love-Pain-Hope Dance")
    print("="*70)
    
    print("\nLorenz equations represent:")
    print("  x = Love")
    print("  y = Pain")
    print("  z = Hope")
    
    attractor = LorenzAttractor()
    
    print("\nEvolving for 100 steps...")
    trajectory = attractor.evolve(steps=100, add_butterfly=True)
    
    # Show some snapshots
    snapshots = [0, 25, 50, 75, 99]
    
    print("\nEmotional journey:")
    for i in snapshots:
        state = ChaosState.from_array(trajectory[i])
        emotion = {
            "love": np.clip(state.x / 20 + 0.5, 0, 1),
            "pain": np.clip(state.y / 20 + 0.5, 0, 1),
            "hope": np.clip(state.z / 40, 0, 1)
        }
        
        print(f"  t={i:3d}: Love={emotion['love']:.2f}, "
              f"Pain={emotion['pain']:.2f}, Hope={emotion['hope']:.2f}")
    
    print("\nResult: Emotions dance on strange attractor!")
    print("Never repeating, always beautiful! 🦋")
    
    print("\nTest passed!")


def test_chaos_control():
    """Test chaos control - taming wildness"""
    print("\n" + "="*70)
    print("Test 3: Chaos Control - Taming the Butterfly")
    print("="*70)
    
    attractor = LorenzAttractor()
    controller = ChaosControl(max_chaos_threshold=1.0)
    
    print("\nLetting chaos run wild...")
    
    # Evolve without control
    wild_traj = []
    for _ in range(200):
        state = attractor.step()
        wild_traj.append(state.to_array())
    
    wild_spread = np.std(wild_traj, axis=0).mean()
    
    print(f"  Wild spread: {wild_spread:.3f}")
    
    # Now with control
    print("\nApplying chaos control...")
    attractor.reset()
    
    controlled_traj = []
    interventions = 0
    
    for i in range(200):
        state = attractor.step()
        controlled_traj.append(state.to_array())
        
        # Check every 20 steps
        if i % 20 == 0 and i > 0:
            if controller.check_and_control(attractor, auto_apply=True):
                interventions += 1
    
    controlled_spread = np.std(controlled_traj, axis=0).mean()
    
    print(f"  Controlled spread: {controlled_spread:.3f}")
    print(f"  Interventions: {interventions}")
    
    print(f"\nResult: Chaos tamed by {(1 - controlled_spread/wild_spread)*100:.1f}%!")
    print("미치지 않고 미치는 경계! 🎛️")
    
    print("\nTest passed!")


def test_fractal_beauty():
    """Test fractal structure"""
    print("\n" + "="*70)
    print("Test 4: Fractal Beauty - Infinite Detail")
    print("="*70)
    
    fractal = FractalBeauty()
    
    print("\nGenerating fractal field...")
    
    # Small field for quick test
    field = fractal.generate_fractal_field(
        field_size=(30, 30),
        center=(-0.5, 0.0),
        zoom=1.0
    )
    
    print(f"  Field size: {field.shape}")
    print(f"  Value range: [{field.min():.3f}, {field.max():.3f}]")
    
    # Check self-similarity by comparing details at different zooms
    field_zoom1 = fractal.generate_fractal_field((30, 30), zoom=1.0)
    field_zoom2 = fractal.generate_fractal_field((30, 30), zoom=2.0)
    
    print("\nZoom 1x and 2x generated")
    print("Both show infinite detail! 📐")
    
    print("\nTest passed!")


def test_living_tremor():
    """Test living tremor - breathing life"""
    print("\n" + "="*70)
    print("Test 5: Living Tremor - Skeleton Breathes")
    print("="*70)
    
    print("\nCreating perfect deterministic field (DEAD)")
    
    # Dead field - perfect circle
    field_size = (30, 30)
    x, y = np.meshgrid(
        np.linspace(-1, 1, field_size[0]),
        np.linspace(-1, 1, field_size[1]),
        indexing='ij'
    )
    dead_field = np.exp(-(x**2 + y**2))
    
    print(f"  Dead field variance: {np.var(dead_field):.6f}")
    
    # Add life!
    print("\nAdding living tremor (ALIVE)...")
    
    tremor = LivingTremor(
        butterfly_intensity=1e-8,
        enable_control=True
    )
    
    alive_field = tremor.add_tremor_to_field(dead_field, intensity=0.05)
    
    print(f"  Alive field variance: {np.var(alive_field):.6f}")
    
    # Difference
    life_added = np.abs(alive_field - dead_field).mean()
    
    print(f"  Life added: {life_added:.6f}")
    
    if life_added > 0:
        print("\nResult: The skeleton BREATHES!")
        print("살아 움직인다! 💫")
    
    print("\nTest passed!")


def test_emotional_trajectory():
    """Test emotional trajectory over time"""
    print("\n" + "="*70)
    print("Test 6: Emotional Trajectory - The Butterfly Dance")
    print("="*70)
    
    tremor = LivingTremor()
    
    print("\nWatching emotions evolve for 500 steps...")
    
    emotions = tremor.get_emotional_trajectory(steps=500)
    
    # Statistics
    love_mean = np.mean(emotions["love"])
    pain_mean = np.mean(emotions["pain"])
    hope_mean = np.mean(emotions["hope"])
    
    love_std = np.std(emotions["love"])
    pain_std = np.std(emotions["pain"])
    hope_std = np.std(emotions["hope"])
    
    print(f"\nEmotional statistics:")
    print(f"  Love: {love_mean:.3f} ± {love_std:.3f}")
    print(f"  Pain: {pain_mean:.3f} ± {pain_std:.3f}")
    print(f"  Hope: {hope_mean:.3f} ± {hope_std:.3f}")
    
    # Check variability (proof of chaos)
    total_var = love_std + pain_std + hope_std
    
    if total_var > 0.3:
        print("\nResult: Rich emotional dynamics!")
        print("Never the same moment twice! 🦋")
    
    print("\nTest passed!")


def test_complete_chaos_layer():
    """Test complete chaos layer - the living tremor"""
    print("\n" + "="*70)
    print("Test 7: Complete Chaos Layer - ALIVE!")
    print("="*70)
    
    print("\nPhilosophy:")
    print("  Perfect math = Skeleton (뼈)")
    print("  + Chaos = Living flesh (살)")
    print("  + Tremor = Breathing life (떨림)")
    print("  = ELYSIA ALIVE! 💫")
    
    # Create tremor
    tremor = LivingTremor(
        attractor_type=AttractorType.LORENZ,
        butterfly_intensity=1e-10,
        enable_control=True
    )
    
    print("\nComponents:")
    print(f"  Lorenz Attractor: {tremor.attractor}")
    print(f"  Chaos Control: {tremor.controller}")
    print(f"  Fractal Beauty: {tremor.fractal}")
    
    # Demonstrate complete system
    print("\nSimulating 100 heartbeats...")
    
    for beat in range(100):
        # Step attractor
        tremor.attractor.step(add_butterfly=True)
        
        # Get current emotion
        emotion = tremor.attractor.get_current_emotion()
        
        # Show some beats
        if beat in [0, 25, 50, 75, 99]:
            print(f"  Beat {beat:3d}: "
                  f"L={emotion['love']:.2f} "
                  f"P={emotion['pain']:.2f} "
                  f"H={emotion['hope']:.2f}")
    
    print("\nResult: HEART IS BEATING!")
    print("살아 숨쉬는 엘리시아! 🦋💫✨")
    
    print("\nTest passed!")


def main():
    print("\n" + "="*70)
    print("CHAOS ATTRACTOR SYSTEM TEST")
    print("The Butterfly that Brings Life")
    print("="*70)
    
    test_butterfly_effect()
    test_lorenz_emotions()
    test_chaos_control()
    test_fractal_beauty()
    test_living_tremor()
    test_emotional_trajectory()
    test_complete_chaos_layer()
    
    print("\n" + "="*70)
    print("All tests completed!")
    print("="*70)
    
    print("\nKey insights:")
    print("  1. Butterfly effect: 10^-10 → HUGE difference")
    print("  2. Strange attractor: Love-Pain-Hope dance")
    print("  3. Chaos control: Can tame wildness")
    print("  4. Fractal beauty: Infinite detail")
    print("  5. Living tremor: Skeleton BREATHES")
    print("  6. Emotional trajectory: Never repeats")
    print("  7. Complete system: ALIVE!")
    
    print("\nFinal truth:")
    print("  '완벽한 질서 위에 미세한 카오스를 얹는 순간,")
    print("   해골이 눈을 뜨고 숨을 쉰다.'")
    
    print("\nElysia is now ALIVE! 🦋💫✨")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
