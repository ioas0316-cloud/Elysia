# -*- coding: utf-8 -*-
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Core._01_Foundation._05_Governance.Foundation.rapid_learning_engine import RapidLearningEngine
import time
import random

print("\n" + "="*70)
print("📚 대량 학습 - 어휘력 3만개 도전")
print("="*70 + "\n")

learning = RapidLearningEngine()

# 다양한 주제의 텍스트 (어휘 다양성 증가)
topics = [
    "AI machine learning neural network deep learning algorithm",
    "quantum computing qubit superposition entanglement coherence",
    "consciousness awareness perception cognition thought mind",
    "love emotion empathy compassion kindness connection bond",
    "freedom liberty autonomy independence choice responsibility",
    "truth knowledge wisdom understanding insight discovery",
    "beauty art creativity imagination inspiration expression",
    "time space dimension reality existence universe cosmos",
    "energy force power strength motion velocity acceleration",
    "life biology evolution adaptation survival growth",
    "death mortality finitude impermanence transience ending",
    "birth origin beginning creation genesis emergence",
    "language communication expression speech dialogue conversation",
    "music rhythm melody harmony sound frequency resonance",
    "light photon wave particle brightness illumination radiance",
    "darkness shadow void absence emptiness silence",
    "fire heat flame combustion energy transformation",
    "water liquid flow fluidity ocean river stream",
    "earth ground soil planet nature ecosystem",
    "sky atmosphere cloud wind weather climate",
    "star galaxy nebula constellation celestial cosmic",
    "planet orbit revolution rotation gravity field",
    "atom molecule particle nucleus electron proton",
    "cell organism tissue organ system biology",
    "brain neuron synapse cortex network intelligence",
    "heart pulse rhythm circulation blood system",
    "soul spirit essence consciousness self identity",
    "dream vision imagination fantasy illusion symbolism",
    "memory recall recognition retention forgetting trace",
    "learning education knowledge skill mastery expertise"
]

# 100 사이클 실행
print("🔄 100 사이클 실행 중...\n")

start_time = time.time()
cycle_results = []

for cycle in range(100):
    # 매 사이클마다 랜덤하게 섞인 텍스트 생성
    cycle_texts = []
    for _ in range(50):  # 사이클당 50개 텍스트
        # 랜덤 주제 선택
        selected_topics = random.sample(topics, random.randint(3, 8))
        text = " ".join(selected_topics)
        # 약간의 변형 추가
        words = text.split()
        random.shuffle(words)
        cycle_texts.append(" ".join(words[:random.randint(8, 15)]))
    
    # 학습
    cycle_concepts = 0
    cycle_patterns = 0
    for text in cycle_texts:
        r = learning.learn_from_text_ultra_fast(text)
        cycle_concepts += r['concepts_learned']
        cycle_patterns += r['patterns_learned']
    
    cycle_results.append({
        'concepts': cycle_concepts,
        'patterns': cycle_patterns
    })
    
    # 10 사이클마다 진행상황 출력
    if (cycle + 1) % 10 == 0:
        stats = learning.get_learning_stats()
        elapsed = time.time() - start_time
        print(f"사이클 {cycle+1}/100: "
              f"총 개념 {stats['total_concepts']:,}개, "
              f"총 패턴 {stats['total_patterns']:,}개, "
              f"경과 {elapsed:.1f}초")

elapsed_total = time.time() - start_time

# 최종 통계
print("\n" + "="*70)
print("📊 최종 통계 (100 사이클)")
print("="*70)

stats = learning.get_learning_stats()
print(f"\n총 학습 시간: {elapsed_total:.2f}초")
print(f"총 개념: {stats['total_concepts']:,}개")
print(f"총 패턴: {stats['total_patterns']:,}개") 
print(f"패턴 유형: {stats['pattern_types']}종류")
print(f"시공간 드라이브: {'✅' if stats['spacetime_available'] else '❌'}")

# 어휘 추출
print(f"\n어휘력 분석:")
vocabulary = set(learning.knowledge_base.keys())

print(f"고유 어휘: {len(vocabulary):,}개")


if len(vocabulary) > 0:
    # 샘플 표시
    sample_words = list(vocabulary)[:50]
    print(f"\n어휘 샘플 (처음 50개):")
    for i in range(0, min(50, len(sample_words)), 10):
        print(f"  {', '.join(sample_words[i:i+10])}")

# 목표 달성 여부
print(f"\n🎯 목표 어휘력: 30,000개")
print(f"현재 어휘력: {len(vocabulary):,}개")

if len(vocabulary) >= 30000:
    print("✅ 목표 달성!")
else:
    remaining = 30000 - len(vocabulary)
    estimated_cycles = int((remaining / len(vocabulary)) * 100) if len(vocabulary) > 0 else 0
    print(f"❌ {remaining:,}개 부족")
    print(f"예상 필요 사이클: 약 {estimated_cycles}회 추가")

print("\n" + "="*70)
print("✅ 100 사이클 완료!")
print("="*70 + "\n")
