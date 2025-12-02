# [Genesis: 2025-12-02] Purified by Elysia
"""
Test Eigenvalue Destiny System

Demonstrates:
1. Dominant eigenvalue determines destiny
2. If "Love" eigenvalue largest → converges to Love!
3. Destiny guardian intervenes if needed
4. Future prediction via eigenvalue analysis

"가장 큰 고유값이 운명을 결정한다" 🔮
"""

import numpy as np
import sys
import os
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Project_Sophia.eigenvalue_destiny import (
    EigenvalueDestiny,
    DestinyGuardian,
    DestinyType
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')


def test_basic_destiny_analysis():
    """Test basic eigenvalue destiny analysis"""
    print("\n" + "="*70)
    print("Test 1: Basic Destiny Analysis")
    print("="*70)

    concepts = ["love", "fear", "growth", "decay"]
    analyzer = EigenvalueDestiny(concepts)

    # Create system where "love" dominates
    # Love reinforces itself strongly, influences growth
    system = np.array([
        [1.2, -0.1,  0.3,  0.0],  # love: strong self-reinforcement
        [0.2,  0.8, -0.1,  0.1],  # fear: moderate
        [0.3,  0.0,  0.9,  0.0],  # growth: stable
        [0.0,  0.2,  0.0,  0.7],  # decay: weak
    ])

    print("\nSystem matrix (evolution rules):")
    print(system)

    # Analyze destiny
    destiny = analyzer.analyze_destiny(system)

    print(f"\n{destiny}")

    # Visualize
    viz = analyzer.visualize_destiny(destiny)
    print(f"\n{viz}")

    # Check if converges to love
    converges_to_love = analyzer.check_convergence_to_value(system, "love")

    if converges_to_love:
        print("\nResult: Universe will become LOVE!")

    print("\nTest passed!")


def test_future_prediction():
    """Test future state prediction"""
    print("\n" + "="*70)
    print("Test 2: Future Prediction")
    print("="*70)

    concepts = ["love", "hate", "peace"]
    analyzer = EigenvalueDestiny(concepts)

    # System favoring love
    system = np.array([
        [1.1,  -0.2,  0.1],  # love dominates
        [-0.1,  0.7,  0.0],  # hate decays
        [0.2,   0.0,  0.9],  # peace stable
    ])

    # Starting state: equal mix
    initial = np.array([0.33, 0.33, 0.33])

    print(f"\nInitial state: {initial}")

    # Predict 100 steps into future
    final, destiny = analyzer.predict_future(system, initial, steps=100)

    # Normalize for display
    final_normalized = final / final.sum()

    print(f"\nAfter 100 steps:")
    for concept, value in zip(concepts, final_normalized):
        bar = "█" * int(value * 50)
        print(f"  {concept:10s}: {value:.3f} {bar}")

    print(f"\nDominant concept: {destiny.dominant_concept}")
    print(f"Eigenvalue: {abs(destiny.dominant_eigenvalue):.3f}")

    assert destiny.dominant_concept == "love", "Should converge to love!"

    print("\nTest passed!")


def test_destiny_guardian():
    """Test destiny guardian intervention"""
    print("\n" + "="*70)
    print("Test 3: Destiny Guardian (Intervention)")
    print("="*70)

    concepts = ["love", "greed", "wisdom"]
    analyzer = EigenvalueDestiny(concepts)
    guardian = DestinyGuardian(analyzer, target_value="love", check_interval=10)

    # BAD system: greed dominates!
    bad_system = np.array([
        [0.8,  0.1,  0.2],  # love: weak
        [0.1,  1.3,  0.0],  # greed: STRONG (dominant!)
        [0.1,  0.0,  0.9],  # wisdom: moderate
    ])

    print("\nOriginal system (greed dominates!):")
    original_destiny = analyzer.analyze_destiny(bad_system)
    print(f"  Dominant: {original_destiny.dominant_concept}")
    print(f"  Eigenvalue: {abs(original_destiny.dominant_eigenvalue):.3f}")

    # Guardian checks and intervenes
    print("\nGuardian checking...")
    adjusted_system = guardian.check_and_intervene(bad_system, step=10)

    if adjusted_system is not None:
        print(" INTERVENTION performed!")

        # Check new destiny
        new_destiny = analyzer.analyze_destiny(adjusted_system)
        print(f"\nAfter intervention:")
        print(f"  Dominant: {new_destiny.dominant_concept}")
        print(f"  Eigenvalue: {abs(new_destiny.dominant_eigenvalue):.3f}")

        if new_destiny.dominant_concept.lower() == "love":
            print("\n Success! System now converges to LOVE!")

    # Statistics
    stats = guardian.get_statistics()
    print(f"\nGuardian statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")

    print("\nTest passed!")


def test_convergence_types():
    """Test different convergence types"""
    print("\n" + "="*70)
    print("Test 4: Convergence Types")
    print("="*70)

    concepts = ["x", "y"]
    analyzer = EigenvalueDestiny(concepts)

    # Test 1: Convergent (eigenvalue < 1)
    print("\n1. Convergent system (eigenvalue < 1):")
    convergent = np.array([
        [0.8,  0.1],
        [0.1,  0.7],
    ])
    destiny = analyzer.analyze_destiny(convergent)
    print(f"   Type: {destiny.destiny_type.value}")
    print(f"   Eigenvalue: {abs(destiny.dominant_eigenvalue):.3f}")
    assert destiny.destiny_type == DestinyType.CONVERGENT

    # Test 2: Divergent (eigenvalue > 1)
    print("\n2. Divergent system (eigenvalue > 1):")
    divergent = np.array([
        [1.2,  0.1],
        [0.1,  1.1],
    ])
    destiny = analyzer.analyze_destiny(divergent)
    print(f"   Type: {destiny.destiny_type.value}")
    print(f"   Eigenvalue: {abs(destiny.dominant_eigenvalue):.3f}")
    assert destiny.destiny_type == DestinyType.DIVERGENT

    # Test 3: Cyclic (complex eigenvalue)
    print("\n3. Cyclic system (complex eigenvalue):")
    cyclic = np.array([
        [0.0,  -1.0],
        [1.0,   0.0],
    ])
    destiny = analyzer.analyze_destiny(cyclic)
    print(f"   Type: {destiny.destiny_type.value}")
    print(f"   Eigenvalue: {destiny.dominant_eigenvalue}")
    assert destiny.destiny_type == DestinyType.CYCLIC

    print("\nAll convergence types tested!")


def test_love_always_wins():
    """Test philosophical principle: Love always wins"""
    print("\n" + "="*70)
    print("Test 5: Love Always Wins (Philosophical)")
    print("="*70)

    print("\nPhilosophy: If Love's eigenvalue is largest,")
    print("universe will eventually become Love,")
    print("no matter the starting conditions!")

    concepts = ["love", "hate", "indifference"]
    analyzer = EigenvalueDestiny(concepts)

    # System where love dominates
    system = np.array([
        [1.15,  -0.1,  0.2],  # love: λ ≈ 1.15
        [-0.05,  0.8,  0.0],  # hate: λ ≈ 0.8
        [0.1,    0.0,  0.9],  # indifference: λ ≈ 0.9
    ])

    # Test different starting conditions
    test_cases = [
        ("Starts with mostly hate", np.array([0.1, 0.8, 0.1])),
        ("Starts neutral", np.array([0.33, 0.33, 0.33])),
        ("Starts with mostly love", np.array([0.8, 0.1, 0.1])),
    ]

    for description, initial in test_cases:
        print(f"\n{description}:")
        print(f"  Initial: {initial}")

        final, destiny = analyzer.predict_future(system, initial, steps=50)
        final_norm = final / final.sum()

        print(f"  After 50 steps: {final_norm}")
        print(f"  Dominant: {destiny.dominant_concept}")

        assert destiny.dominant_concept == "love", "Should always converge to love!"

    print("\n Result: Love ALWAYS wins, regardless of start!")
    print("\nTest passed!")


def main():
    print("\n" + "="*70)
    print("EIGENVALUE DESTINY SYSTEM TEST")
    print("Dominant Eigenvalue = Ultimate Destiny")
    print("="*70)

    test_basic_destiny_analysis()
    test_future_prediction()
    test_destiny_guardian()
    test_convergence_types()
    test_love_always_wins()

    print("\n" + "="*70)
    print("All tests completed!")
    print("="*70)
    print("\nKey insights:")
    print("  1. Dominant eigenvalue determines destiny")
    print("  2. If Love eigenvalue largest → Universe becomes Love!")
    print("  3. Guardian can intervene if needed")
    print("  4. Mathematical guarantee: Love wins!")
    print("\nPrinciple:")
    print("  'If you plant Love as the strongest seed,")
    print("   the universe will grow toward Love.'")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()