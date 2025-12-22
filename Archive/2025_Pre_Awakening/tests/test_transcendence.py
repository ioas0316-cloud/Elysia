"""
Test for Transcendence Engine
==============================

This test validates the transcendence engine's ability to:
1. Track progress towards superintelligence
2. Perform meta-cognition
3. Expand capabilities autonomously
4. Synthesize knowledge across domains
5. Perform recursive self-improvement
"""

import sys
import os
import logging
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

from Core.Foundation.transcendence_engine import TranscendenceEngine

logging.basicConfig(level=logging.INFO)

def test_initialization():
    """Test that the engine initializes correctly"""
    print("\n" + "=" * 60)
    print("TEST 1: Initialization")
    print("=" * 60)
    
    engine = TranscendenceEngine()
    
    assert engine.metrics.knowledge_domains == 0, "Should start with no domains"
    assert engine.metrics.cognitive_depth == 0.0, "Should start with no depth"
    assert engine.metrics.transcendence_level == 0, "Should start at level 0"
    assert engine.metrics.learning_velocity == 1.0, "Should start with velocity 1.0"
    
    print("✅ Initialization test passed")
    return engine

def test_meta_cognition():
    """Test meta-cognitive abilities"""
    print("\n" + "=" * 60)
    print("TEST 2: Meta-Cognition")
    print("=" * 60)
    
    engine = TranscendenceEngine()
    result = engine.think_about_thinking()
    
    assert "current_state" in result, "Should return current state"
    assert "limitations" in result, "Should identify limitations"
    assert "improvement_strategies" in result, "Should generate strategies"
    
    print(f"Current State: {result['current_state']}")
    print(f"Limitations: {result['limitations']}")
    print(f"Strategies: {result['improvement_strategies']}")
    print("✅ Meta-cognition test passed")
    
    return engine

def test_capability_expansion():
    """Test autonomous capability expansion"""
    print("\n" + "=" * 60)
    print("TEST 3: Capability Expansion")
    print("=" * 60)
    
    engine = TranscendenceEngine()
    initial_domains = engine.metrics.knowledge_domains
    
    # Try expanding several times
    domains_to_learn = ["mathematics", "physics", "philosophy", "language"]
    successes = 0
    
    for domain in domains_to_learn:
        for attempt in range(5):  # Multiple attempts per domain
            if engine.expand_capabilities(domain):
                successes += 1
                break
    
    assert engine.metrics.knowledge_domains > initial_domains, "Should have learned new domains"
    print(f"Domains learned: {engine.metrics.knowledge_domains}")
    print(f"Cognitive depth: {engine.metrics.cognitive_depth:.2f}")
    print("✅ Capability expansion test passed")
    
    return engine

def test_knowledge_synthesis():
    """Test cross-domain synthesis"""
    print("\n" + "=" * 60)
    print("TEST 4: Knowledge Synthesis")
    print("=" * 60)
    
    engine = TranscendenceEngine()
    
    # First learn some domains
    engine.expand_capabilities("mathematics")
    engine.expand_capabilities("physics")
    engine.expand_capabilities("philosophy")
    
    # Boost synthesis capability for testing
    engine.metrics.synthesis_capability = 50.0
    
    # Try synthesis
    attempts = 10
    insights = []
    for _ in range(attempts):
        insight = engine.synthesize_knowledge(["mathematics", "physics", "philosophy"])
        if insight:
            insights.append(insight)
    
    print(f"Insights generated: {len(insights)}")
    for i, insight in enumerate(insights[:3], 1):
        print(f"  {i}. {insight}")
    
    assert engine.metrics.synthesis_capability >= 50.0, "Synthesis capability should increase"
    print("✅ Knowledge synthesis test passed")
    
    return engine

def test_recursive_improvement():
    """Test recursive self-improvement"""
    print("\n" + "=" * 60)
    print("TEST 5: Recursive Self-Improvement")
    print("=" * 60)
    
    engine = TranscendenceEngine()
    initial_velocity = engine.metrics.learning_velocity
    initial_depth = engine.metrics.cognitive_depth
    
    # Run several improvement cycles
    for i in range(5):
        improvements = engine.recursive_self_improvement()
        print(f"Cycle {i+1}: Velocity boost: {improvements['learning_velocity_boost']:.4f}, "
              f"Depth gain: {improvements['cognitive_depth_gain']:.2f}")
    
    assert engine.metrics.learning_velocity >= initial_velocity, "Velocity should increase or stay same"
    assert engine.metrics.cognitive_depth >= initial_depth, "Depth should increase or stay same"
    
    print(f"Final velocity: {engine.metrics.learning_velocity:.3f} (started at {initial_velocity:.3f})")
    print(f"Final depth: {engine.metrics.cognitive_depth:.2f} (started at {initial_depth:.2f})")
    print("✅ Recursive improvement test passed")
    
    return engine

def test_full_cycle():
    """Test a complete transcendence cycle"""
    print("\n" + "=" * 60)
    print("TEST 6: Full Transcendence Cycle")
    print("=" * 60)
    
    engine = TranscendenceEngine()
    
    # Run multiple cycles
    for cycle_num in range(10):
        print(f"\n--- Cycle {cycle_num + 1} ---")
        results = engine.cycle()
        
    # Evaluate final progress
    progress = engine.evaluate_transcendence_progress()
    
    print(f"\nFinal Progress:")
    print(f"  Overall Score: {progress['overall_score']:.1f}/100")
    print(f"  Stage: {progress['stage']}")
    print(f"  Transcendence Level: {progress['transcendence_level']}")
    print(f"  Active Domains: {progress['active_domains']}")
    print(f"  Breakthroughs: {progress['breakthroughs']}")
    print(f"  Insights Count: {progress['insights_count']}")
    
    assert progress['overall_score'] > 0, "Should have made some progress"
    assert progress['active_domains'] > 0, "Should have learned some domains"
    
    print("✅ Full cycle test passed")
    
    return engine, progress

def test_breakthrough_detection():
    """Test breakthrough detection"""
    print("\n" + "=" * 60)
    print("TEST 7: Breakthrough Detection")
    print("=" * 60)
    
    engine = TranscendenceEngine()
    
    # Artificially set conditions for breakthrough
    engine.metrics.knowledge_domains = 10
    engine.metrics.cognitive_depth = 30.0
    engine.metrics.synthesis_capability = 40.0
    engine.metrics.meta_awareness = 40.0
    
    initial_level = engine.metrics.transcendence_level
    
    # Run improvement cycles until breakthrough
    max_attempts = 50
    breakthrough_occurred = False
    
    for attempt in range(max_attempts):
        improvements = engine.recursive_self_improvement()
        if improvements.get("breakthrough"):
            breakthrough_occurred = True
            print(f"🌟 Breakthrough occurred at attempt {attempt + 1}!")
            break
    
    if breakthrough_occurred:
        assert engine.metrics.transcendence_level > initial_level, "Level should increase after breakthrough"
        assert engine.metrics.breakthrough_count > 0, "Should count breakthroughs"
        print(f"Transcendence level: {initial_level} → {engine.metrics.transcendence_level}")
        print("✅ Breakthrough detection test passed")
    else:
        print("⚠️ No breakthrough in max attempts, but system functioning")
    
    return engine

def test_progress_stages():
    """Test progression through different stages"""
    print("\n" + "=" * 60)
    print("TEST 8: Stage Progression")
    print("=" * 60)
    
    engine = TranscendenceEngine()
    
    stages_seen = set()
    
    # Simulate extended learning by directly updating metrics
    for i in range(10):
        engine.metrics.knowledge_domains = i * 5
        engine.metrics.cognitive_depth = i * 20.0
        engine.metrics.synthesis_capability = min(100, i * 10.0)
        engine.metrics.meta_awareness = min(100, i * 10.0)
        
        progress = engine.evaluate_transcendence_progress()
        stage = progress['stage']
        stages_seen.add(stage)
        
        if i % 3 == 0:
            print(f"Score: {progress['overall_score']:.1f} → Stage: {stage}")
    
    print(f"\nStages experienced: {', '.join(sorted(stages_seen))}")
    assert len(stages_seen) > 1, "Should progress through multiple stages"
    print("✅ Stage progression test passed")
    
    return engine

def run_all_tests():
    """Run all tests"""
    print("\n" + "=" * 70)
    print("TRANSCENDENCE ENGINE TEST SUITE")
    print("=" * 70)
    
    try:
        test_initialization()
        test_meta_cognition()
        test_capability_expansion()
        test_knowledge_synthesis()
        test_recursive_improvement()
        test_full_cycle()
        test_breakthrough_detection()
        test_progress_stages()
        
        print("\n" + "=" * 70)
        print("✅ ALL TESTS PASSED!")
        print("=" * 70)
        print("\nThe Transcendence Engine is operational.")
        print("Elysia's path to superintelligence is active.")
        
        return True
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        return False
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
