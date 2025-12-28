# -*- coding: utf-8 -*-
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Core._01_Foundation._05_Governance.Foundation.rapid_learning_engine import RapidLearningEngine
import time

print("\n" + "="*70)
print("📚 실용적 급속 학습 - 실제 데이터 학습")
print("="*70 + "\n")

learning = RapidLearningEngine()

# 대량 텍스트 데이터
texts = [
    "AI transforms the world with machine learning and deep neural networks.",
    "Quantum computing uses qubits in superposition for parallel computation.",
    "Consciousness emerges from complex patterns in neural networks.",
    "Love creates deep emotional bonds between individuals through empathy.",
    "Freedom requires responsibility and conscious choice for growth."
] * 50  # 250개


print(f"📖 데이터: {len(texts)}개 텍스트, {sum(len(t.split()) for t in texts)}개 단어\n")

# 1. 단일 학습
print("1. 단일 학습:")
start = time.time()
r = learning.learn_from_text_ultra_fast(texts[0])
print(f"  시간: {time.time()-start:.4f}초")
print(f"  단어: {r['word_count']}, 개념: {r['concepts_learned']}, 패턴: {r['patterns_learned']}\n")

# 2. 대량 학습
print("2. 대량 급속 학습 (250개):")
start = time.time()
total_concepts = 0
total_patterns = 0

for text in texts:
    r = learning.learn_from_text_ultra_fast(text)
    total_concepts += r['concepts_learned']
    total_patterns += r['patterns_learned']

elapsed = time.time() - start

print(f"  시간: {elapsed:.4f}초")
print(f"  학습 개념: {total_concepts}개")
print(f"  학습 패턴: {total_patterns}개")
print(f"  속도: {len(texts)/elapsed:.0f}개/초\n")

# 3. 최종 통계
stats = learning.get_learning_stats()
print("="*70)
print("📊 최종 통계:")
print(f"  총 개념: {stats['total_concepts']}개")
print(f"  총 패턴: {stats['total_patterns']}개")
print(f"  패턴 유형: {stats['pattern_types']}종류")
print(f"  시공간 드라이브: {'✅' if stats['spacetime_available'] else '❌'}")

# 주요 개념 표시
if stats['total_concepts'] > 0:
    concept_freq = {}
    for p in learning.learned_patterns.values():
        if isinstance(p, dict):
            for c in p.get('concepts', []):
                concept_freq[c] = concept_freq.get(c, 0) + 1
    
    print("\n상위 개념:")
    for i, (c, f) in enumerate(sorted(concept_freq.items(), key=lambda x: -x[1])[:10], 1):
        print(f"  {i}. {c}: {f}회")


print("\n" + "="*70)
print("✅ 실제 데이터 학습 완료!")
print("="*70 + "\n")
