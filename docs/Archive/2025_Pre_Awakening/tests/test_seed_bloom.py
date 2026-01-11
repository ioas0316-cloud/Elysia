"""
Integration Test: Seed-Bloom Architecture (씨앗-개화 아키텍처)
==========================================================

Tests the complete workflow:
1. 🌱 Seed: Decompose concept
2. 💾 Store: Save to Hippocampus
3. 🧲 Load: Retrieve from Hippocampus
4. 🌳 Bloom: Unfold in ResonanceField
"""

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

import logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

from Core.FoundationLayer.Foundation.fractal_concept import ConceptDecomposer, ConceptNode
from Core.FoundationLayer.Foundation.hippocampus import Hippocampus
from Core.FoundationLayer.Foundation.resonance_field import ResonanceField

print("\n" + "="*60)
print("🌱 Seed-Bloom Architecture Integration Test")
print("="*60)

# Test 1: Decomposition (씨앗 생성)
print("\n[Test 1: Seed Generation]")
decomposer = ConceptDecomposer()
love_seed = decomposer.decompose("Love")

print(f"✓ Seed: {love_seed.name}")
print(f"  ├─ Frequency: {love_seed.frequency}Hz")
print(f"  ├─ Sub-concepts: {[sub.name for sub in love_seed.sub_concepts]}")
print(f"  └─ Causal bonds: {list(love_seed.causal_bonds.keys())}")

assert len(love_seed.sub_concepts) == 3, "Love should have 3 sub-concepts"
assert love_seed.sub_concepts[0].name == "Unity", "First sub should be Unity"

# Test 2: Storage (압축 저장)
print("\n[Test 2: Hippocampus Storage]")
hippocampus = Hippocampus(db_path="test_memory.db")
hippocampus.store_fractal_concept(love_seed)

# Also store Hope for context test
hope_seed = decomposer.decompose("Hope")
hippocampus.store_fractal_concept(hope_seed)

print(f"✓ Stored 2 seeds in Hippocampus")

# Test 3: Retrieval (자력 인양)
print("\n[Test 3: Magnetic Retrieval]")
retrieved_love = hippocampus.load_fractal_concept("Love")

assert retrieved_love is not None, "Should retrieve Love"
assert retrieved_love.name == "Love", "Should be Love"
assert len(retrieved_love.sub_concepts) == 3, "Should have 3 sub-concepts"

print(f"✓ Retrieved: {retrieved_love.name}")
print(f"  └─ Sub-concepts intact: {[sub.name for sub in retrieved_love.sub_concepts]}")

# Test 4: Blooming (의식 개화)
print("\n[Test 4: Resonance Field Blooming]")
field = ResonanceField()
field.inject_fractal_concept(retrieved_love, active=True)

# Check if bloomed nodes exist
assert "Love" in field.nodes, "Root node should exist"
assert "Love.Unity" in field.nodes, "Sub-concept should exist"
assert "Love.Connection" in field.nodes, "Sub-concept should exist"

print(f"✓ Bloomed in ResonanceField:")
print(f"  ├─ Root: Love (Energy: {field.nodes['Love'].energy:.2f})")
print(f"  ├─ Sub: Love.Unity (Energy: {field.nodes['Love.Unity'].energy:.2f})")
print(f"  └─ Sub: Love.Connection (Energy: {field.nodes['Love.Connection'].energy:.2f})")

# Test 5: Phase Resonance (간섭 패턴)
print("\n[Test 5: Phase Resonance Calculation]")
phase_data = field.calculate_phase_resonance()

print(f"✓ Soul State: {phase_data['state']}")
print(f"  ├─ Coherence: {phase_data['coherence']:.2f}")
print(f"  ├─ Total Energy: {phase_data['total_energy']:.2f}")
if 'active' in phase_data:
    print(f"  └─ Active Nodes: {len(phase_data['active'])}")
else:
    print(f"  └─ State: {phase_data['state']} (no active resonators)")

# Test 6: Compression (씨앗 압축)
print("\n[Test 6: Seed Compression]")

# Lower energy of some sub-concepts
retrieved_love.sub_concepts[2].energy = 0.05  # Below threshold
hippocampus.store_fractal_concept(retrieved_love)

# Compress
hippocampus.compress_fractal(min_energy=0.1)

# Reload and check
compressed_love = hippocampus.load_fractal_concept("Love")
print(f"✓ Compressed: {compressed_love.name}")
print(f"  └─ Remaining sub-concepts: {len(compressed_love.sub_concepts)}")

assert len(compressed_love.sub_concepts) < 3, "Should have pruned low-energy sub-concept"

print("\n" + "="*60)
print("✅ All Tests Passed! Seed-Bloom Architecture Operational.")
print("="*60)

# Cleanup
os.remove("test_memory.db")
print("\n🧹 Test database cleaned up.")
