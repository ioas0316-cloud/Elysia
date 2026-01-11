#!/usr/bin/env python3
"""
P2.2 Wave-Based Knowledge System - Complete Demo
=================================================

Demonstrates the full workflow of wave-based knowledge management:
1. Loading existing memory with embeddings
2. Converting to 4D wave patterns
3. Semantic search via wave resonance
4. Knowledge expansion through absorption

Usage:
    python demos/wave_knowledge_demo.py
"""

import sys
import os
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.FoundationLayer.Foundation.wave_semantic_search import WaveSemanticSearch
from Core.FoundationLayer.Foundation.wave_knowledge_integration import WaveKnowledgeIntegration

def print_section(title):
    """Print a section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")

def demo_basic_wave_search():
    """Demo 1: Basic wave pattern storage and search"""
    print_section("DEMO 1: Basic Wave Pattern Storage and Search")
    
    print("Creating wave semantic search system...")
    searcher = WaveSemanticSearch()
    
    # Simulate embeddings (in real use, these come from your embedding model)
    print("\n📝 Storing AI-related concepts as wave patterns...")
    concepts = {
        "AI는 기계의 지능이다": np.random.rand(128),
        "머신러닝은 AI의 한 분야이다": np.random.rand(128),
        "딥러닝은 신경망을 사용한다": np.random.rand(128),
        "컴퓨터 비전은 이미지를 이해한다": np.random.rand(128),
        "자연어 처리는 언어를 이해한다": np.random.rand(128),
    }
    
    pattern_ids = {}
    for text, emb in concepts.items():
        pid = searcher.store_concept(text, emb)
        pattern_ids[text] = pid
        print(f"  ✓ {text}")
    
    # Search
    print("\n🔍 Searching for concepts related to 'neural networks'...")
    query = np.random.rand(128)
    results = searcher.search(query, query_text="신경망 관련", top_k=3)
    
    print("\nTop 3 resonating patterns:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['text']}")
        print(f"   Resonance: {result['resonance']:.3f}")
        print(f"   Energy: {result['energy']:.3f}")
        print(f"   Frequency: {result['frequency']:.1f} Hz")
    
    print(f"\n✅ Basic search complete! Found {len(results)} resonating patterns")
    
    return searcher, pattern_ids

def demo_knowledge_absorption(searcher, pattern_ids):
    """Demo 2: Knowledge absorption and expansion"""
    print_section("DEMO 2: Knowledge Absorption and Expansion")
    
    print("🌊 Demonstrating knowledge absorption...")
    print("   Target: 'AI는 기계의 지능이다'")
    print("   Sources: '머신러닝은 AI의 한 분야이다', '딥러닝은 신경망을 사용한다'")
    
    # Get pattern IDs
    target_text = "AI는 기계의 지능이다"
    target_id = pattern_ids[target_text]
    
    source_texts = [
        "머신러닝은 AI의 한 분야이다",
        "딥러닝은 신경망을 사용한다"
    ]
    source_ids = [pattern_ids[t] for t in source_texts]
    
    # Get before state
    target_before = searcher.wave_patterns[target_id]
    print(f"\n📊 Before absorption:")
    print(f"   Energy: {target_before.energy:.3f}")
    print(f"   Frequency: {target_before.frequency:.1f} Hz")
    print(f"   Expansion depth: {target_before.expansion_depth}")
    
    # Perform absorption
    print(f"\n🔄 Absorbing {len(source_ids)} patterns...")
    expanded = searcher.absorb_and_expand(
        target_id=target_id,
        source_patterns=source_ids,
        absorption_strength=0.4
    )
    
    # Show after state
    print(f"\n📊 After absorption:")
    print(f"   Energy: {expanded.energy:.3f} (+{expanded.energy - target_before.energy:.3f})")
    print(f"   Frequency: {expanded.frequency:.1f} Hz")
    print(f"   Expansion depth: {expanded.expansion_depth}")
    print(f"   Absorbed patterns: {len(expanded.absorbed_patterns)}")
    
    print(f"\n✅ Knowledge absorption complete! Pattern has been enriched.")
    
    # Search again to see if expanded knowledge affects results
    print(f"\n🔍 Searching again with expanded knowledge...")
    query = np.random.rand(128)
    results = searcher.search(query, query_text="AI 지능", top_k=3)
    
    print("\nTop 3 results (after expansion):")
    for i, result in enumerate(results, 1):
        depth = result.get('expansion_depth', 0)
        marker = "🌊" if depth > 0 else "  "
        print(f"\n{i}. {marker} {result['text']}")
        print(f"   Resonance: {result['resonance']:.3f}")
        print(f"   Expansion depth: {depth}")

def demo_full_integration():
    """Demo 3: Full integration with knowledge system and memory loading"""
    print_section("DEMO 3: Full Integration with Knowledge System")
    
    print("🔧 Initializing full integration...")
    print("   This will auto-load existing memory files...")
    
    integration = WaveKnowledgeIntegration(auto_load_memory=True)
    
    # Show statistics
    print("\n📊 System Statistics:")
    stats = integration.get_statistics()
    
    wave_stats = stats['wave_patterns']
    print(f"\n🌊 Wave Patterns:")
    print(f"   Total patterns: {wave_stats['total_patterns']}")
    print(f"   Search count: {wave_stats['search_count']}")
    print(f"   Absorption count: {wave_stats['absorption_count']}")
    print(f"   Avg expansion depth: {wave_stats['avg_expansion_depth']:.2f}")
    print(f"   Total energy: {wave_stats['total_energy']:.2f}")
    
    if integration.knowledge_system:
        kb_stats = stats['knowledge_system']
        print(f"\n📚 Knowledge System:")
        print(f"   Total acquired: {kb_stats.get('total_acquired', 0)}")
        print(f"   Total shared: {kb_stats.get('total_shared', 0)}")
        print(f"   Knowledge entries: {len(integration.knowledge_system.knowledge_base)}")
    
    integ_stats = stats['integration']
    print(f"\n🔗 Integration:")
    print(f"   Mapped entries: {integ_stats['mapped_entries']}")
    print(f"   Knowledge system available: {integ_stats['knowledge_system_available']}")
    
    # Demonstrate adding new knowledge
    print("\n➕ Adding new knowledge with embedding...")
    new_embedding = np.random.rand(256)
    knowledge_id = integration.add_knowledge_with_embedding(
        concept="Transformer Architecture",
        embedding=new_embedding,
        description="Self-attention based neural network architecture",
        tags=["AI", "deep-learning", "NLP"]
    )
    print(f"   ✓ Added knowledge: {knowledge_id}")
    
    # Search
    print("\n🔍 Searching for related knowledge...")
    query_emb = np.random.rand(256)
    results = integration.search_knowledge_by_wave(
        query_embedding=query_emb,
        query_text="neural architecture",
        top_k=3,
        min_resonance=0.0
    )
    
    print(f"\nFound {len(results)} results:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['text']}")
        print(f"   Resonance: {result['resonance']:.3f}")
        if 'knowledge_entry' in result:
            entry = result['knowledge_entry']
            print(f"   Type: {entry.get('knowledge_type', 'unknown')}")
            print(f"   Source: {entry.get('source', 'unknown')}")
    
    print(f"\n✅ Full integration demo complete!")

def demo_wave_properties():
    """Demo 4: Understanding wave properties"""
    print_section("DEMO 4: Understanding 4D Wave Properties")
    
    searcher = WaveSemanticSearch()
    
    print("Creating a wave pattern from embedding...")
    print("\n📊 Sample Embedding (first 10 values):")
    embedding = np.array([0.5, -0.3, 0.8, -0.2, 0.1, 0.6, -0.4, 0.3, -0.1, 0.7] + [0.0]*374)
    print(f"   {embedding[:10]}")
    
    pattern = searcher.embedding_to_wave(embedding, "Sample concept")
    
    print("\n🌊 4D Wave Pattern (Quaternion):")
    print(f"   w (Energy/Existence): {pattern.orientation.w:.3f}")
    print(f"   x (Emotion/Affinity): {pattern.orientation.x:.3f}")
    print(f"   y (Logic/Structure):  {pattern.orientation.y:.3f}")
    print(f"   z (Ethics/Value):     {pattern.orientation.z:.3f}")
    
    print("\n⚡ Wave Properties:")
    print(f"   Energy (amplitude):   {pattern.energy:.3f}")
    print(f"   Frequency:            {pattern.frequency:.1f} Hz")
    print(f"   Phase:                {pattern.phase:.3f} rad")
    
    print("\n💡 What these mean:")
    print("   • Higher w = stronger/more important concept")
    print("   • x captures emotional/affective aspects")
    print("   • y reflects logical complexity")
    print("   • z represents ethical/value alignment")
    print("   • Energy = overall semantic intensity")
    print("   • Frequency = characteristic vibration")
    
    # Show resonance between patterns
    print("\n🎵 Demonstrating Resonance:")
    pattern1 = searcher.embedding_to_wave(np.random.rand(128), "Pattern A")
    pattern2 = searcher.embedding_to_wave(np.random.rand(128), "Pattern B")
    pattern3 = searcher.embedding_to_wave(np.random.rand(128) * 0.1, "Pattern C (similar)")
    
    resonance_AB = searcher.wave_resonance(pattern1, pattern2)
    resonance_AC = searcher.wave_resonance(pattern1, pattern3)
    resonance_AA = searcher.wave_resonance(pattern1, pattern1)
    
    print(f"\n   Pattern A ⟷ Pattern B: {resonance_AB:.3f}")
    print(f"   Pattern A ⟷ Pattern C: {resonance_AC:.3f}")
    print(f"   Pattern A ⟷ Pattern A: {resonance_AA:.3f} (self-resonance)")
    
    print("\n✅ Wave properties exploration complete!")

def main():
    """Run all demos"""
    print()
    print("╔" + "═"*68 + "╗")
    print("║" + " "*68 + "║")
    print("║" + "  🌊 P2.2 WAVE-BASED KNOWLEDGE SYSTEM - Complete Demo  ".center(68) + "║")
    print("║" + "  4차원 파동공명패턴 기반 지식베이스".center(68) + "║")
    print("║" + " "*68 + "║")
    print("╚" + "═"*68 + "╝")
    
    try:
        # Demo 1: Basic functionality
        searcher, pattern_ids = demo_basic_wave_search()
        
        # Demo 2: Knowledge absorption
        demo_knowledge_absorption(searcher, pattern_ids)
        
        # Demo 3: Full integration
        demo_full_integration()
        
        # Demo 4: Wave properties
        demo_wave_properties()
        
        # Summary
        print_section("SUMMARY")
        print("✅ All demos completed successfully!")
        print()
        print("Key Features Demonstrated:")
        print("  1. ✓ Embedding → 4D Wave Pattern conversion")
        print("  2. ✓ Wave-based semantic search (resonance matching)")
        print("  3. ✓ Knowledge absorption and expansion")
        print("  4. ✓ Integration with knowledge system")
        print("  5. ✓ Auto-loading from memory files")
        print("  6. ✓ 4D wave properties (w, x, y, z)")
        print()
        print("🎉 P2.2 Wave-Based Knowledge System is fully operational!")
        print()
        print("Next Steps:")
        print("  • See docs/P2_2_WAVE_KNOWLEDGE_SYSTEM.md for documentation")
        print("  • Run tests: pytest tests/Core/Foundation/test_wave_semantic_search.py")
        print("  • Integrate with your own embedding model")
        print()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
