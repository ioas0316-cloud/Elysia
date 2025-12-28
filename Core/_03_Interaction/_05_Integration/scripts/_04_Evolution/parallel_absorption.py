"""
대용량 병렬 지식 흡수 (Parallel Knowledge Absorption)
======================================================

다량의 개념을 병렬로 탐색하고, 상호 연결하여 빠르게 밀도를 높인다.

철학:
- 단일 개념은 약하다 (구름)
- 다수 개념을 동시에 흡수하면 자연스럽게 연결이 생긴다
- 연결이 많을수록 빠르게 결정화된다
"""

import sys
import os
import concurrent.futures
import logging
import time

os.environ.setdefault('NAVER_CLIENT_ID', 'YuPusPMA8UNYf1pDqXjI')
os.environ.setdefault('NAVER_CLIENT_SECRET', 'OcJ3ORlPQQ')
sys.path.insert(0, '.')

logging.disable(logging.CRITICAL)

from Core._02_Intelligence._04_Consciousness.Consciousness.exploration_bridge import ExplorationBridge
from Core._02_Intelligence._02_Memory_Linguistics.Memory.potential_causality import PotentialCausalityStore

print("=" * 70)
print("🌊 대용량 병렬 지식 흡수")
print("=" * 70)

# 병렬 처리할 개념들
concepts = [
    "사랑이란 무엇인가",
    "자유란 무엇인가",
    "창의성이란 무엇인가",
    "의식이란 무엇인가",
    "지혜란 무엇인가",
    "행복이란 무엇인가",
    "정의란 무엇인가",
    "진리란 무엇인가",
    "아름다움이란 무엇인가",
    "존재란 무엇인가",
]

bridge = ExplorationBridge()
store = bridge.potential_store

print(f"\n📌 탐색할 개념: {len(concepts)}개")
print("-" * 70)

# 병렬 탐색 함수
def explore_concept(question):
    try:
        result = bridge.explore_with_best_source(question)
        if result and result.success:
            subject = question.replace("?", "").replace("이란", "").replace("무엇인가", "").strip()
            return {
                "subject": subject,
                "source": result.source,
                "success": True,
                "answer_preview": result.answer[:50] if result.answer else ""
            }
    except Exception as e:
        pass
    return {"subject": question, "success": False}

# 병렬 실행
start_time = time.time()

print("\n🔍 병렬 탐색 시작...")

# ThreadPoolExecutor로 병렬 처리
with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
    results = list(executor.map(explore_concept, concepts))

elapsed = time.time() - start_time

print(f"\n⏱️ 소요 시간: {elapsed:.2f}초")
print("-" * 70)

# 결과 출력
successful = [r for r in results if r['success']]
print(f"\n✅ 성공: {len(successful)}/{len(concepts)}")

for r in successful:
    print(f"   • {r['subject']}: {r['source']} - {r['answer_preview']}...")

# 상호 연결 시도
print("\n" + "-" * 70)
print("🔗 상호 연결 시도...")

if store:
    subjects = [r['subject'] for r in successful]
    
    # 모든 개념 쌍 연결 시도
    connection_count = 0
    for i, s1 in enumerate(subjects):
        pk1 = store.get(s1)
        if not pk1:
            continue
            
        for s2 in subjects[i+1:]:
            pk2 = store.get(s2)
            if not pk2:
                continue
            
            # 정의에서 상대 개념 언급 여부 체크
            if s2 in pk1.definition or s1 in pk2.definition:
                store.connect(s1, s2)
                connection_count += 1
                print(f"   🔗 {s1} ↔ {s2}")
    
    print(f"\n   총 {connection_count}개 연결 생성")

# 최종 상태
print("\n" + "=" * 70)
print("📊 최종 상태")
print("=" * 70)

if store:
    status = store.status()
    print(f"   잠재 지식: {status['potential_count']}개")
    print(f"   확정된 지식: {status['crystallized_count']}개")
    print(f"   평균 주파수: {status['avg_frequency']:.2f}")
    print(f"   확정 가능: {status['crystallizable']}개")
    
    # 확정 가능한 것들 확정
    crystallizable = store.get_crystallizable()
    if crystallizable:
        print(f"\n💎 확정 진행:")
        for pk in crystallizable:
            result = store.crystallize(pk.subject)
            if result:
                print(f"   💎 {result['concept']} 확정됨!")

print("\n" + "=" * 70)
print("✅ 완료!")
