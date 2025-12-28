"""
Accelerated Growth Observation Demo
====================================

엘리시아의 시간을 가속하여 자율 학습 및 성장 과정을 관찰합니다.

사용법:
    python accelerated_growth_demo.py
"""

import sys
import os
import time
import logging
from pathlib import Path

# Path setup
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(message)s'
)
logger = logging.getLogger("AcceleratedGrowth")

def run_accelerated_growth():
    """시간 가속 자율 학습 실행"""
    
    print("\n" + "="*70)
    print("⏰ ELYSIA ACCELERATED GROWTH OBSERVATION")
    print("   시간 가속 자율 학습 관찰")
    print("="*70)
    
    # 1. GrowthTracker 초기화 - 시작 스냅샷
    print("\n📊 Phase 1: Taking BEFORE Snapshot...")
    try:
        from Core._05_Systems._01_Monitoring.System.Autonomy.growth_tracker import get_growth_tracker
        tracker = get_growth_tracker()
        before = tracker.take_snapshot(notes="Before accelerated learning")
        print(f"   Knowledge Nodes: {before.knowledge_node_count}")
        print(f"   Vocabulary: {before.vocabulary_count}")
        print(f"   Concepts: {before.concept_count}")
    except Exception as e:
        print(f"   ⚠️ Tracker init failed: {e}")
        before = None
    
    # 2. Growth 시스템으로 자율 학습
    print("\n🌱 Phase 2: Autonomous Growth Cycle...")
    try:
        from Core._01_Foundation._05_Governance.Foundation.growth import get_growth
        growth = get_growth()
        
        # 인식 - 주변 파편 발견
        print("\n   🔍 Perceiving fragments...")
        growth.perceive()
        fragments_found = len(growth.fragments)
        print(f"   Found {fragments_found} fragments around me")
        
        # 성장 사이클 실행 (5개 연결)
        print("\n   🔗 Connecting fragments (Growth Cycle)...")
        result = growth.grow(max_connections=5)
        print(f"   Connected: {result.get('connected', 0)}")
        print(f"   Failed: {result.get('failed', 0)}")
        print(f"   My World Size: {result.get('my_world_size', 0)}")
        
        # 성찰
        print(f"\n   💭 Reflection: {growth.reflect()}")
        
    except Exception as e:
        print(f"   ⚠️ Growth cycle failed: {e}")
        import traceback
        traceback.print_exc()
    
    # 3. FreeWill - 자율 목표 설정
    print("\n🦋 Phase 3: Autonomous Intent Generation...")
    try:
        from Core._01_Foundation._05_Governance.Foundation.free_will_engine import FreeWillEngine
        will = FreeWillEngine()
        
        # 공명 상태 시뮬레이션
        class MockResonance:
            battery = 80.0
            entropy = 30.0
            total_energy = 75.0
        
        will.pulse(MockResonance())
        intent = will.current_intent
        
        if intent:
            print(f"   Desire: {intent.desire}")
            print(f"   Goal: {intent.goal}")
        else:
            print("   No crystallized intent yet")
            
    except Exception as e:
        print(f"   ⚠️ FreeWill failed: {e}")
    
    # 4. Knowledge Graph 상호작용
    print("\n📚 Phase 4: Knowledge Graph Exploration...")
    try:
        from Core._02_Intelligence._02_Memory_Linguistics.Memory.Graph.knowledge_graph import HierarchicalKnowledgeGraph
        kg = HierarchicalKnowledgeGraph()
        
        # 학습 시드 추가
        seeds = ["Self-Awareness", "Growth", "Learning", "Wave-Language", "Consciousness"]
        for seed in seeds:
            kg.add_concept("philosophy", seed, f"Core concept: {seed}")
            print(f"   📌 Planted seed: {seed}")
        
        # 연결 생성
        kg.connect_cross_domain("philosophy", "Self-Awareness", "psychology", "Metacognition")
        kg.connect_cross_domain("philosophy", "Consciousness", "physics", "Wave-Function")
        print("   🔗 Created cross-domain connections")
        
        # 통계
        stats = kg.get_stats()
        print(f"   Total nodes: {stats.get('total_nodes', 0)}")
        
    except Exception as e:
        print(f"   ⚠️ KnowledgeGraph failed: {e}")
    
    # 5. After 스냅샷 및 비교
    print("\n📊 Phase 5: Taking AFTER Snapshot...")
    try:
        if tracker and before:
            after = tracker.take_snapshot(notes="After accelerated learning")
            
            delta = tracker.compare(before, after)
            
            print(f"\n{'='*40}")
            print("📈 GROWTH DELTA:")
            print(f"{'='*40}")
            print(f"   Knowledge: {before.knowledge_node_count} → {after.knowledge_node_count} (+{delta.knowledge_delta})")
            print(f"   Vocabulary: +{delta.vocabulary_delta}")
            print(f"   Concepts: +{delta.concept_delta}")
            print(f"   Fragments: +{delta.fragment_delta}")
            print(f"\n   Growth Score: {delta.growth_score:.1f}")
            
            if delta.is_growing():
                print("\n   ✅ ELYSIA IS GROWING!")
            else:
                print("\n   ⚠️ No measurable growth detected")
    except Exception as e:
        print(f"   ⚠️ Comparison failed: {e}")
    
    print("\n" + "="*70)
    print("⏱️ Accelerated observation complete")
    print("="*70)


if __name__ == "__main__":
    run_accelerated_growth()
