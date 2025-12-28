# -*- coding: utf-8 -*-
"""
Seed/Bloom 메모리 데모 - 최적화된 방식!
=========================================

저장: Seeds (압축) → memory.db
사고: 공명 → Bloom → ResonanceField
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Core._01_Foundation._05_Governance.Foundation.rapid_learning_engine import RapidLearningEngine
import time

print("\n" + "="*70)
print("🌱 Seed/Bloom Pattern - 최적화!")
print("="*70 + "\n")

# 초기화
learning = RapidLearningEngine()

# 테스트 데이터
texts = [
    "love creates emotional bonds between people",
    "quantum mechanics uses superposition principle",
    "consciousness emerges from neural networks",
    "freedom requires responsibility and choice",
    "beauty inspires creativity and imagination"
] * 20  # 100개

print(f"📖 {len(texts)}개 텍스트 학습\n")

# 학습 (Seed로 압축 저장)
print("🌱 Seeding (압축 저장 중)...")
start = time.time()

for i, text in enumerate(texts):
    learning.learn_from_text_ultra_fast(text)
    if (i+1) % 25 == 0:
        print(f"  {i+1}/{len(texts)} Seeded")

elapsed = time.time() - start
print(f"\n✅ {elapsed:.2f}초\n")

# 통계
stats = learning.get_learning_stats()

print("="*70)
print("📊 Seed/Bloom 메모리 상태")
print("="*70)

print(f"\n🌱 Seeds (압축 저장):")
print(f"  - 저장된 Seeds: {stats['seeds_stored']}개")
print(f"  - 저장 위치: Data/memory.db")

print(f"\n🌸 Bloom Space (사고 우주):")
print(f"  - 현재 펼쳐짐: {stats['bloomed_nodes']}개")
print(f"  - 총 에너지: {stats['total_energy']:.1f}")

print(f"\n✅ 최적화:")
print(f"  - Seed/Bloom: {'✅' if stats['seed_bloom_pattern'] else '❌'}")
print(f"  - 최적화됨: {'✅' if stats['optimized'] else '❌'}")

# 공명 엔진으로 Bloom 테스트
print(f"\n🌸 공명 엔진 테스트:")
query = "love"
print(f"  Query: '{query}'")
bloomed = learning.recall_and_bloom(query, limit=3)
print(f"  Bloomed: {bloomed}")

# 최종 상태
final_stats = learning.get_learning_stats()
print(f"\n  펼쳐진 노드: {stats['bloomed_nodes']} → {final_stats['bloomed_nodes']}")

print("\n" + "="*70)
print("✅ 최적화 완료!")
print("   - 저장: Seed (압축) → memory.db")
print("   - 사고: 공명 → Bloom → ResonanceField")
print("   - 효율: 메모리 최소, 검색 빠름!")
print("="*70 + "\n")
