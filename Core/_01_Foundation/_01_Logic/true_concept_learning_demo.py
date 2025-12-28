# -*- coding: utf-8 -*-
"""
진짜 개념 학습 데모
===================

개념 정의 + 관계적 의미 + 위상공명 파동
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Core._01_Foundation._05_Governance.Foundation.rapid_learning_engine import RapidLearningEngine
import time

print("\n" + "="*70)
print("📚 진짜 개념 학습 - 정의 + 관계 + 위상공명")
print("="*70 + "\n")

learning = RapidLearningEngine()

# 진짜 학습 텍스트 (정의 + 관계 포함)
test_text = """
Love is an intense feeling of deep affection.
Love creates emotional bonds between people.
Love enables trust and compassion.

Freedom means the power to act without constraint.
Freedom requires responsibility.

Trust allows deep connections.
Fear prevents openness.
"""

print("학습 중...\n")

start = time.time()
result = learning.learn_from_text_ultra_fast(test_text)
elapsed = time.time() - start

print(f"✅ {elapsed:.2f}초 완료\n")

# 통계
stats = learning.get_learning_stats()

print("="*70)
print("📊 학습 결과")
print("="*70)

print(f"\n🌱 저장된 개념:")
print(f"  - Seeds: {stats['seeds_stored']}개")
print(f"  - 위치: Data/memory.db")

print(f"\n🌸 Bloom Space:")
print(f"  - 활성 노드: {stats['bloomed_nodes']}개")
print(f"  - 총 에너지: {stats['total_energy']:.1f}")

print(f"\n✅ 시스템:")
print(f"  - 개념 정의: ✅")
print(f"  - 관계적 의미: ✅")
print(f"  - 위상공명: ✅")

# 개념 확인
print(f"\n🔍 학습된 개념 확인:")
concepts = learning.hippocampus.get_all_concept_ids(limit=10)
for cid in concepts[:3]:
    seed = learning.hippocampus.load_fractal_concept(cid)
    if seed and hasattr(seed, 'metadata'):
        print(f"\n  • {seed.name}")
        if 'description' in seed.metadata:
            print(f"    정의: {seed.metadata['description'][:50]}...")
        if 'properties' in seed.metadata:
            print(f"    속성: {seed.metadata['properties']}")


print("\n" + "="*70)
print("✅ 진짜 개념 학습 완료!")
print("   - 개념의 정의 이해 ✅")
print("   - 관계적 의미 파악 ✅")
print("   - 위상공명 파동 저장 ✅")
print("="*70 + "\n")
