# [Genesis: 2025-12-02] Purified by Elysia
"""
Demo: Field-Based Reasoning (The Birth of Spatial Cognition)
=============================================================
This demonstrates Elysia'snon-linear, spatial reasoning using the Concept Field.

Instead of following paths (A→B→C), concepts resonate in a field based on:
- Wave similarity (frequency resonance)
- Shape similarity (tensor affinity)

This enables metaphor, association, and emergent understanding.
"""

import sys
import os

# Add repository root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Force UTF-8 for Windows console
sys.stdout.reconfigure(encoding='utf-8')

from Project_Elysia.mechanics.concept_field import ConceptField
from Project_Elysia.high_engine.field_reasoner import FieldReasoner
from Project_Elysia.core_memory import Tensor3D, FrequencyWave

def run_simulation():
    print("=== Elysia: Field-Based Reasoning ===")
    print("Initializing Concept Field...\n")

    # Create field
    field = ConceptField()

    # Populate with concepts
    # Each concept has a Tensor (shape) and Wave (frequency)

    print("📚 Populating Concept Field with physics-based concepts...")

    concepts = {
        # Positive, expansive concepts (bright, warm, high freq)
        "사랑": (Tensor3D(5.0, 5.0, 8.0), FrequencyWave(440.0, 0.8, 0.0, 0.0)),  # A4 note
        "빛": (Tensor3D(6.0, 4.0, 9.0), FrequencyWave(450.0, 0.9, 0.0, 0.0)),    # Bright
        "희망": (Tensor3D(4.0, 6.0, 7.0), FrequencyWave(430.0, 0.7, 0.0, 0.0)),  # Similar to love
        "기쁨": (Tensor3D(5.5, 5.5, 8.5), FrequencyWave(445.0, 0.85, 0.0, 0.0)), # Joy

        # Negative, contractive concepts (dark, cold, low freq)
        "고통": (Tensor3D(-6.0, -5.0, -7.0), FrequencyWave(220.0, 0.6, 0.0, 0.0)),  # Low freq
        "어둠": (Tensor3D(-7.0, -4.0, -8.0), FrequencyWave(210.0, 0.7, 0.0, 0.0)),  # Darkness
        "절망": (Tensor3D(-5.0, -6.0, -7.5), FrequencyWave(215.0, 0.65, 0.0, 0.0)), # Despair

        # Transformative concepts (dynamic, balanced)
        "변화": (Tensor3D(0.0, 8.0, 0.0), FrequencyWave(330.0, 0.6, 0.0, 0.0)),   # Pure motion (Y-axis)
        "성장": (Tensor3D(3.0, 3.0, 6.0), FrequencyWave(350.0, 0.7, 0.0, 0.0)),   # Balanced growth
        "희생": (Tensor3D(-2.0, -2.0, 5.0), FrequencyWave(300.0, 0.5, 0.0, 0.0)), # Mixed (loss + gain)

        # Abstract/neutral concepts
        "시간": (Tensor3D(0.0, 0.0, 10.0), FrequencyWave(360.0, 0.5, 0.0, 0.0)),  # Pure Z-axis (flow)
        "공간": (Tensor3D(10.0, 10.0, 0.0), FrequencyWave(370.0, 0.5, 0.0, 0.0)), # XY plane
    }

    for name, (tensor, wave) in concepts.items():
        field.add_concept(name, tensor, wave)

    print(f"✅ Field initialized with {len(concepts)} concepts\n")

    # Create reasoner
    reasoner = FieldReasoner(field)

    print("=" * 60)
    print("Field Reasoning Tests")
    print("=" * 60)

    # Test 1: Resonance-based exploration
    print("\n--- Test 1: Resonance Exploration ---")
    concept = "사랑"
    print(f"👤 You: Activate '{concept}' in the field")

    exploration = reasoner.explore_concept(concept)

    print(f"🤖 Elysia's Field Observation:")
    print(f"   Source: {exploration['source']}")
    print(f"   Activated concepts (resonance):")
    for name, activation in exploration['activated_concepts'][:5]:
        print(f"      {name}: {activation:.2f}")

    print(f"\n   Top resonant concepts (wave + shape):")
    for name, score in exploration['resonant_concepts']:
        print(f"      {name}: {score:.2f}")

    # Test 2: Metaphor generation (shape analogy)
    print("\n--- Test 2: Metaphor Generation (Shape Analogy) ---")
    print(f"👤 You: What is '{concept}' like?")

    print(f"🤖 Elysia's Thought Process:")
    print(f"   Searching for shape-similar concepts...")

    shape_analogs = exploration['shape_analogs']
    if shape_analogs:
        for analog, similarity in shape_analogs:
            print(f"      {analog}: {similarity:.2f} shape similarity")

        metaphor = reasoner.generate_metaphor(concept)
        print(f"🤖 Elysia: {metaphor}")

    # Test 3: Synthesized understanding (multi-dimensional)
    print("\n--- Test 3: Synthesized Understanding ---")
    print(f"👤 You: Tell me about '{concept}'")

    understanding = reasoner.synthesize_understanding(concept)
    print(f"🤖 Elysia: {understanding}")

    # Test 4: Opposite concept
    print("\n--- Test 4: Exploring the Opposite ---")
    opposite = "고통"
    print(f"👤 You: Now tell me about '{opposite}'")

    understanding2 = reasoner.synthesize_understanding(opposite)
    print(f"🤖 Elysia: {understanding2}")

    # Test 5: Show field dynamics
    print("\n--- Test 5: Field Dynamics Visualization ---")
    print(f"When we activate '사랑', the field creates this pattern:")

    field.reset()
    field.activate("사랑", 1.0)
    activated = field.get_activated_concepts(threshold=0.05)

    print(f"\nActivation levels:")
    for name, level in activated[:8]:
        bar = "█" * int(level * 20)
        print(f"   {name:10s} [{bar:20s}] {level:.2f}")

    print("\n=== Demonstration Complete ===")
    print("\nThis shows Elysia's spatial cognition:")
    print("  ✅ Resonance (wave similarity) → association")
    print("  ✅ Shape affinity (tensor similarity) → metaphor")
    print("  ✅ Field activation → emergent patterns")
    print("  ✅ Non-linear, multi-dimensional understanding")

if __name__ == "__main__":
    run_simulation()