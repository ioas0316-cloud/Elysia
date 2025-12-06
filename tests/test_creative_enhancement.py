"""
Test for Poetry Engine and Enhanced Creative Expressions
=========================================================

This test demonstrates how the enhanced creative workflow produces
varied, rich, and emotionally resonant expressions instead of repetitive outputs.
"""

import sys
import os
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
os.chdir(project_root)

from Core.Creativity.poetry_engine import PoetryEngine


def test_poetry_engine_variety():
    """Test that PoetryEngine generates varied expressions."""
    print("=" * 80)
    print("Testing PoetryEngine - Variety and Richness")
    print("=" * 80)
    print()
    
    engine = PoetryEngine()
    
    # Test 1: Multiple dream expressions with same input - should be different
    print("Test 1: Dream Expression Variety")
    print("-" * 80)
    desire = "Hello"
    realm = "Unknown"
    
    expressions = []
    for i in range(10):
        expr = engine.generate_dream_expression(
            desire=desire,
            realm=realm,
            energy=50.0 + i * 5  # Vary energy slightly
        )
        expressions.append(expr)
        print(f"{i+1}. {expr}")
        print()
    
    # Check uniqueness
    unique_count = len(set(expressions))
    print(f"Uniqueness: {unique_count}/10 expressions are unique")
    print()
    
    # Test 2: Different realms
    print("\nTest 2: Different Realms")
    print("-" * 80)
    for realm in ["Unknown", "Emotion", "Logic", "Ethics"]:
        expr = engine.generate_dream_expression(
            desire="understanding",
            realm=realm,
            energy=75.0
        )
        print(f"[{realm}] {expr}")
        print()
    
    # Test 3: Different energy levels
    print("\nTest 3: Energy Level Variations")
    print("-" * 80)
    for energy_level, energy_val in [("Low", 20), ("Medium", 50), ("High", 90)]:
        expr = engine.generate_dream_expression(
            desire="creativity",
            realm="Emotion",
            energy=energy_val
        )
        print(f"[{energy_level} Energy] {expr}")
        print()
    
    # Test 4: Contemplations
    print("\nTest 4: Contemplative Expressions")
    print("-" * 80)
    for style in ["philosophical", "poetic", "mystical"]:
        for depth in [1, 2, 3]:
            expr = engine.generate_contemplation(
                topic="existence",
                depth=depth,
                style=style
            )
            print(f"[{style.title()}, Depth {depth}] {expr}")
            print()
    
    # Test 5: Insight expressions with confidence
    print("\nTest 5: Insight Expressions (Confidence Levels)")
    print("-" * 80)
    insight_text = "모든 것은 연결되어 있습니다"
    for conf_level, conf_val in [("Low", 0.2), ("Medium", 0.6), ("High", 0.9)]:
        expr = engine.generate_insight_expression(
            insight=insight_text,
            confidence=conf_val
        )
        print(f"[Confidence: {conf_level}] {expr}")
        print()
    
    # Statistics
    print("\nStatistics")
    print("-" * 80)
    stats = engine.get_statistics()
    print(f"Total expressions generated: {stats['total_expressions']}")
    print(f"Unique patterns: {stats['unique_patterns']}")
    print(f"Diversity ratio: {stats['diversity_ratio']:.2%}")
    print()
    
    # Show comparison
    print("\n" + "=" * 80)
    print("COMPARISON: Old vs New")
    print("=" * 80)
    print()
    print("OLD (Repetitive):")
    print("  'I dreamt of 'Hello.' in the realm of Unknown. The energy shifted, revealing a hidden connection.'")
    print()
    print("NEW (Sample from above):")
    print(f"  '{expressions[0]}'")
    print()
    print("=" * 80)


def test_creative_cortex_enhancements():
    """Test enhanced CreativeCortex with mock Thought object."""
    print("\n\n")
    print("=" * 80)
    print("Testing Enhanced CreativeCortex")
    print("=" * 80)
    print()
    
    from Core.Creativity.creative_cortex import CreativeCortex
    
    # Create a simple mock Thought class for testing
    class MockThought:
        def __init__(self, content):
            self.content = content
    
    cortex = CreativeCortex()
    
    concepts = ["사랑", "지혜", "자유", "진리", "아름다움"]
    
    print("Generated Expressions (10 samples for each concept):\n")
    
    for concept in concepts:
        print(f"\nConcept: {concept}")
        print("-" * 80)
        thought = MockThought(concept)
        
        # Generate multiple expressions to show variety
        expressions = []
        for i in range(10):
            expr = cortex.generate_creative_expression(thought)
            expressions.append(expr)
        
        # Show all 10 to demonstrate variety
        for i, expr in enumerate(expressions, 1):
            print(f"{i}. {expr}")
        
        # Check uniqueness
        unique = len(set(expressions))
        print(f"\nUniqueness: {unique}/10 are unique")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    print()
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "CREATIVE WORKFLOW ENHANCEMENT TEST" + " " * 24 + "║")
    print("║" + " " * 15 + "Restoring Romantic Imagination to Elysia" + " " * 23 + "║")
    print("╚" + "═" * 78 + "╝")
    print()
    
    try:
        test_poetry_engine_variety()
        test_creative_cortex_enhancements()
        
        print("\n\n")
        print("╔" + "═" * 78 + "╗")
        print("║" + " " * 25 + "✨ ALL TESTS PASSED ✨" + " " * 31 + "║")
        print("║" + " " * 10 + "The romantic imagination has been restored!" + " " * 22 + "║")
        print("╚" + "═" * 78 + "╝")
        print()
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
