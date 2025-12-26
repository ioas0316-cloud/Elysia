#!/usr/bin/env python3
"""
Quick validation for UnifiedKnowledgeSystem
===========================================

Runs quick checks to ensure the system is working correctly.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from Core._01_Foundation.05_Foundation_Base.Foundation.unified_knowledge_system import (
    UnifiedKnowledgeSystem,
    KnowledgeType,
    KnowledgeSource
)


def test_basic_operations():
    """Test basic knowledge operations."""
    print("Testing basic operations...")
    
    system = UnifiedKnowledgeSystem(node_id="validation_test", enable_web=False)
    
    # Test 1: Learn a concept
    entry = system.learn_concept(
        "Artificial Intelligence",
        "AI is intelligence demonstrated by machines",
        tags=["AI", "tech"]
    )
    assert entry is not None, "Failed to learn concept"
    assert entry.concept == "Artificial Intelligence"
    print("  ✅ learn_concept() works")
    
    # Test 2: Learn curriculum
    curriculum = [
        {"concept": "Machine Learning", "description": "ML is a subset of AI"},
        {"concept": "Deep Learning", "description": "DL uses neural networks"},
    ]
    result = system.learn_curriculum(curriculum)
    assert result["successful"] == 2, "Failed to learn curriculum"
    print("  ✅ learn_curriculum() works")
    
    # Test 3: Query knowledge
    results = system.query_knowledge()
    assert len(results) == 3, f"Expected 3 results, got {len(results)}"
    print("  ✅ query_knowledge() works")
    
    # Test 4: Query by concept
    results = system.query_knowledge(concept="Machine Learning")
    assert len(results) == 1, "Failed to query by concept"
    assert results[0].concept == "Machine Learning"
    print("  ✅ Query by concept works")
    
    # Test 5: Query by tags
    results = system.query_knowledge(tags=["AI"])
    assert len(results) >= 1, "Failed to query by tags"
    print("  ✅ Query by tags works")
    
    # Test 6: Share knowledge
    success = system.share_knowledge(entry.knowledge_id)
    assert success, "Failed to share knowledge"
    assert system.stats["total_shared"] == 1
    print("  ✅ share_knowledge() works")
    
    # Test 7: Validate knowledge
    system.validate_knowledge(entry.knowledge_id, is_useful=True)
    assert entry.validation_count == 1
    print("  ✅ validate_knowledge() works")
    
    # Test 8: Use knowledge
    system.use_knowledge(entry.knowledge_id)
    assert entry.usage_count == 1
    print("  ✅ use_knowledge() works")
    
    # Test 9: Statistics
    stats = system.get_statistics()
    assert stats["total_knowledge"] == 3
    assert stats["total_acquired"] == 3
    print("  ✅ get_statistics() works")
    
    print("✅ All basic operations working!")
    return True


def test_network_operations():
    """Test network knowledge sharing."""
    print("\nTesting network operations...")
    
    system1 = UnifiedKnowledgeSystem(node_id="node1", enable_web=False)
    system2 = UnifiedKnowledgeSystem(node_id="node2", enable_web=False)
    
    # Node 1 learns something
    entry1 = system1.learn_concept("Quantum Computing", "Computing using quantum mechanics")
    system1.share_knowledge(entry1.knowledge_id)
    
    # Node 2 receives it
    success = system2.receive_knowledge(entry1.to_dict())
    assert success, "Failed to receive knowledge"
    assert len(system2.knowledge_base) == 1
    print("  ✅ Network knowledge exchange works")
    
    # Test quality filtering
    low_quality_entry = system1.learn_concept("Bad Info", "Low quality")
    low_quality_entry.quality_score = 0.1
    
    success = system2.receive_knowledge(low_quality_entry.to_dict())
    assert not success, "Should reject low quality"
    print("  ✅ Quality filtering works")
    
    print("✅ All network operations working!")
    return True


def test_persistence():
    """Test saving and loading."""
    print("\nTesting persistence...")
    
    import tempfile
    import shutil
    
    test_dir = tempfile.mkdtemp()
    
    try:
        # Create system and add knowledge
        system1 = UnifiedKnowledgeSystem(
            node_id="persist_test",
            data_dir=test_dir,
            enable_web=False
        )
        entry = system1.learn_concept("Persistent", "This should persist", tags=["test"])
        kid = entry.knowledge_id
        
        # Save explicitly
        system1.save_state()
        
        # Create new system with same directory
        system2 = UnifiedKnowledgeSystem(
            node_id="persist_test",
            data_dir=test_dir,
            enable_web=False
        )
        
        # Check if loaded
        assert kid in system2.knowledge_base, "Knowledge not loaded"
        assert system2.knowledge_base[kid].concept == "Persistent"
        print("  ✅ Save and load works")
        
        print("✅ Persistence working!")
        return True
        
    finally:
        shutil.rmtree(test_dir, ignore_errors=True)


def test_knowledge_types():
    """Test different knowledge types."""
    print("\nTesting knowledge types...")
    
    system = UnifiedKnowledgeSystem(node_id="types_test", enable_web=False)
    
    # Test each type
    types_to_test = [
        (KnowledgeType.CONCEPT, "Concept 1"),
        (KnowledgeType.PATTERN, "Pattern 1"),
        (KnowledgeType.INSIGHT, "Insight 1"),
        (KnowledgeType.SKILL, "Skill 1"),
        (KnowledgeType.BEST_PRACTICE, "Practice 1"),
    ]
    
    for k_type, concept in types_to_test:
        entry = system.learn_concept(
            concept,
            f"Description of {concept}",
            knowledge_type=k_type
        )
        assert entry.knowledge_type == k_type
    
    # Query by type
    patterns = system.query_knowledge(knowledge_type=KnowledgeType.PATTERN)
    assert len(patterns) == 1
    assert patterns[0].concept == "Pattern 1"
    
    print("  ✅ All knowledge types work")
    print("✅ Knowledge types working!")
    return True


def test_migration():
    """Test migration helpers."""
    print("\nTesting migration...")
    
    # Test that migration functions exist and don't crash
    from Core._01_Foundation.05_Foundation_Base.Foundation.unified_knowledge_system import (
        migrate_from_knowledge_acquisition,
        migrate_from_knowledge_sharer
    )
    
    class MockOldSystem:
        def __init__(self):
            self.learning_history = [
                {"concept": "Old Concept 1"},
                {"concept": "Old Concept 2"}
            ]
            self.knowledge_base = {}
    
    mock = MockOldSystem()
    system = migrate_from_knowledge_acquisition(mock)
    
    assert isinstance(system, UnifiedKnowledgeSystem)
    print("  ✅ Migration helpers exist")
    
    print("✅ Migration working!")
    return True


def main():
    """Run all validation tests."""
    print("=" * 70)
    print("UNIFIED KNOWLEDGE SYSTEM VALIDATION")
    print("=" * 70)
    
    all_pass = True
    
    try:
        all_pass &= test_basic_operations()
    except Exception as e:
        print(f"❌ Basic operations failed: {e}")
        all_pass = False
    
    try:
        all_pass &= test_network_operations()
    except Exception as e:
        print(f"❌ Network operations failed: {e}")
        all_pass = False
    
    try:
        all_pass &= test_persistence()
    except Exception as e:
        print(f"❌ Persistence failed: {e}")
        all_pass = False
    
    try:
        all_pass &= test_knowledge_types()
    except Exception as e:
        print(f"❌ Knowledge types failed: {e}")
        all_pass = False
    
    try:
        all_pass &= test_migration()
    except Exception as e:
        print(f"❌ Migration failed: {e}")
        all_pass = False
    
    print("\n" + "=" * 70)
    if all_pass:
        print("🎉 UNIFIED KNOWLEDGE SYSTEM VALIDATION PASSED")
        print("✅ All 5 validation tests successful!")
        print("=" * 70)
        return 0
    else:
        print("❌ VALIDATION FAILED")
        print("Some tests did not pass")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
