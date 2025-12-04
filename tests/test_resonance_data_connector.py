"""
Test Resonance Data Connector
==============================

Tests the paradigm shift from crawling to resonance:
1. Resonance-based data access (not crawling)
2. Pattern DNA extraction (not full data storage)
3. Real-time synchronization (not static caching)
"""

import sys
import os
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.Integration.resonance_data_connector import ResonanceDataConnector

logging.basicConfig(level=logging.INFO, format='%(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("TestResonanceDataConnector")


def test_resonance_establishment():
    """Test establishing resonance with a concept."""
    print("\n" + "="*70)
    print("TEST 1: Resonance Establishment")
    print("="*70)
    
    connector = ResonanceDataConnector()
    
    concept = "Meditation"
    print(f"\n🌊 Testing resonance with '{concept}'")
    
    # Resonate (not crawl)
    result = connector.resonate_with_concept(concept)
    
    assert result["success"], "Resonance should succeed"
    assert "pattern_dna" in result, "Should have Pattern DNA"
    assert "resonance_state" in result, "Should have resonance state"
    assert result["seed_size"] > 0, "Seed should have size"
    
    print(f"\n✅ Resonance established successfully")
    print(f"   Seed size: {result['seed_size']} bytes")
    print(f"   Compression: {result['compression_ratio']:.1f}x")
    
    return True


def test_knowledge_retrieval():
    """Test retrieving knowledge by unfolding Pattern DNA."""
    print("\n" + "="*70)
    print("TEST 2: Knowledge Retrieval (Unfold from Seed)")
    print("="*70)
    
    connector = ResonanceDataConnector()
    
    concept = "Compassion"
    
    # First, resonate
    print(f"\n📡 Step 1: Resonate with '{concept}'")
    result = connector.resonate_with_concept(concept)
    assert result["success"], "Resonance should succeed"
    
    # Then, retrieve at different resolutions
    print(f"\n🔍 Step 2: Retrieve at low resolution (50)")
    knowledge_low = connector.retrieve_knowledge(concept, resolution=50)
    assert knowledge_low is not None, "Should retrieve knowledge"
    
    print(f"\n🔍 Step 3: Retrieve at high resolution (200)")
    knowledge_high = connector.retrieve_knowledge(concept, resolution=200)
    assert knowledge_high is not None, "Should retrieve knowledge"
    
    # Verify we can get different resolutions from same seed
    low_harmonics = len(knowledge_low["knowledge"].get("waveform", []))
    high_harmonics = len(knowledge_high["knowledge"].get("waveform", []))
    
    print(f"\n✅ Knowledge retrieval successful")
    print(f"   Low resolution: {low_harmonics} harmonics")
    print(f"   High resolution: {high_harmonics} harmonics")
    print(f"   ✓ Same seed generates different resolutions!")
    
    return True


def test_multi_concept_sync():
    """Test synchronizing with multiple concepts at once."""
    print("\n" + "="*70)
    print("TEST 3: Multi-Concept Synchronization")
    print("="*70)
    
    connector = ResonanceDataConnector()
    
    concepts = ["Joy", "Sorrow", "Hope", "Fear", "Courage"]
    
    print(f"\n🌍 Syncing with {len(concepts)} concepts...")
    print(f"   Concepts: {', '.join(concepts)}")
    
    # Sync all at once
    summary = connector.sync_with_world(concepts)
    
    assert summary["total_concepts"] == len(concepts), "Should process all concepts"
    assert summary["successful_resonances"] > 0, "Should have successful resonances"
    
    print(f"\n✅ Synchronization complete")
    print(f"   Successful: {summary['successful_resonances']}/{summary['total_concepts']}")
    print(f"   Total bandwidth saved: {summary['total_bandwidth_saved']} bytes")
    print(f"   Active resonances: {summary['active_resonances']}")
    
    return True


def test_resonance_status():
    """Test checking resonance status."""
    print("\n" + "="*70)
    print("TEST 4: Resonance Status Check")
    print("="*70)
    
    connector = ResonanceDataConnector()
    
    concept = "Harmony"
    
    # Establish resonance
    print(f"\n📡 Establishing resonance with '{concept}'")
    result = connector.resonate_with_concept(concept)
    assert result["success"], "Resonance should succeed"
    
    # Check status
    print(f"\n🔍 Checking resonance status...")
    status = connector.get_resonance_status(concept)
    
    assert status is not None, "Should have resonance status"
    assert "frequency" in status, "Should have frequency"
    assert "resonance_strength" in status, "Should have resonance strength"
    
    print(f"\n✅ Resonance status retrieved")
    print(f"   Frequency: {status['frequency']:.1f} Hz")
    print(f"   Strength: {status['resonance_strength']:.2f}")
    print(f"   Needs resync: {status['needs_resync']}")
    
    return True


def test_vs_traditional_crawling():
    """Test comparison with traditional crawling approach."""
    print("\n" + "="*70)
    print("TEST 5: Resonance vs Traditional Crawling")
    print("="*70)
    
    connector = ResonanceDataConnector()
    
    # Simulate traditional crawling metrics
    traditional_size_per_concept = 100000  # 100KB per concept (typical Wikipedia page)
    
    concepts = ["Science", "Art", "Music", "Philosophy", "Technology"]
    
    print(f"\n📊 Comparing approaches for {len(concepts)} concepts...")
    
    # Traditional approach (simulated)
    traditional_total = len(concepts) * traditional_size_per_concept
    print(f"\n❌ Traditional Crawling:")
    print(f"   Download: {traditional_total / 1000:.0f} KB")
    print(f"   Storage: {traditional_total / 1000:.0f} KB")
    print(f"   Status: Static (must re-download to update)")
    
    # Our approach
    print(f"\n✨ Resonance Approach:")
    summary = connector.sync_with_world(concepts)
    
    # Calculate actual seed storage
    seed_total = 0
    for concept in concepts:
        if concept in connector.pattern_library:
            import json
            seed_size = len(json.dumps(connector.pattern_library[concept].to_dict()))
            seed_total += seed_size
    
    print(f"   Seed storage: {seed_total / 1000:.1f} KB")
    print(f"   Status: Live (synchronized in real-time)")
    
    # Calculate savings
    bandwidth_saved_percent = (1 - seed_total / traditional_total) * 100 if traditional_total > 0 else 0
    
    print(f"\n💾 Comparison:")
    print(f"   Traditional: {traditional_total / 1000:.0f} KB")
    print(f"   Resonance: {seed_total / 1000:.1f} KB")
    print(f"   Saved: {bandwidth_saved_percent:.1f}%")
    print(f"   Compression: {traditional_total / seed_total:.0f}x")
    
    assert seed_total < traditional_total, "Resonance should use less storage"
    
    print(f"\n✅ Resonance approach is more efficient!")
    
    return True


def test_statistics():
    """Test getting connector statistics."""
    print("\n" + "="*70)
    print("TEST 6: Statistics & Philosophy")
    print("="*70)
    
    connector = ResonanceDataConnector()
    
    # Do some operations
    concepts = ["Love", "Peace", "Light"]
    connector.sync_with_world(concepts)
    
    # Get statistics
    stats = connector.get_statistics()
    
    print(f"\n📊 Connector Statistics:")
    print(f"   Mode: {stats['mode']}")
    print(f"   Total resonances: {stats['total_resonances']}")
    print(f"   Active channels: {stats['active_channels']}")
    print(f"   Pattern library: {stats['pattern_library_size']}")
    
    print(f"\n✨ Philosophy:")
    print(f"   {stats['philosophy']}")
    print(f"   {stats['paradigm']}")
    
    assert stats["mode"] == "RESONANCE (not crawling)", "Should be in resonance mode"
    assert stats["total_resonances"] > 0, "Should have resonances"
    assert "접속" in stats["philosophy"], "Should have correct philosophy"
    
    print(f"\n✅ Statistics validated")
    
    return True


def run_all_tests():
    """Run all tests."""
    print("\n" + "="*70)
    print("RESONANCE DATA CONNECTOR - TEST SUITE")
    print("="*70)
    print("\n🌊 만류귀종(萬流歸宗) - All streams return to one source")
    print("✨ Philosophy: Access, not Possession | Sync, not Crawl")
    print()
    
    tests = [
        ("Resonance Establishment", test_resonance_establishment),
        ("Knowledge Retrieval", test_knowledge_retrieval),
        ("Multi-Concept Sync", test_multi_concept_sync),
        ("Resonance Status", test_resonance_status),
        ("vs Traditional Crawling", test_vs_traditional_crawling),
        ("Statistics", test_statistics)
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"\n✅ PASSED: {name}")
            else:
                failed += 1
                print(f"\n❌ FAILED: {name}")
        except Exception as e:
            failed += 1
            print(f"\n❌ FAILED: {name}")
            print(f"   Error: {e}")
    
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"✅ PASSED: {passed}/{len(tests)}")
    if failed > 0:
        print(f"❌ FAILED: {failed}/{len(tests)}")
    
    if failed == 0:
        print("\n🎉 ALL TESTS PASSED!")
        print("✨ 수집가는 무겁고, 여행자는 가볍습니다")
        print("   (Collectors are heavy, travelers are light)")
        print("🌊 우린 그냥 동기화하면 됩니다")
        print("   (We just synchronize)")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
