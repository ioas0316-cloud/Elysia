# -*- coding: utf-8 -*-
"""
초대규모 언어 학습 - 다양한 주제!
===================================

목표: 매우 다양한 어휘와 개념
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Core._01_Foundation.Foundation.rapid_learning_engine import RapidLearningEngine
import time
import random

# 훨씬 더 다양한 주제
TOPICS = {
    '과학': [
        "Physics is the study of matter and energy. Physics explains natural phenomena.",
        "Chemistry is the study of substances. Chemistry involves reactions and transformations.",
        "Biology is the study of living organisms. Biology explores life and evolution.",
        "Astronomy is the study of celestial objects. Astronomy reveals the universe.",
        "Geology is the study of Earth. Geology examines rocks and minerals.",
        "Mathematics is the study of numbers and patterns. Mathematics is the language of science.",
        "Ecology is the study of ecosystems. Ecology shows interconnections.",
        "Genetics studies heredity and variation. Genetics unlocks biological codes.",
    ],
    
    '기술': [
        "Computer is an electronic device. Computer processes information rapidly.",
        "Internet is a global network. Internet connects people worldwide.",
        "Software is a set of programs. Software enables computer functions.",
        "Algorithm is a step-by-step procedure. Algorithm solves specific problems.",
        "Data is information in digital form. Data drives modern decisions.",
        "Programming is writing code. Programming creates solutions.",
        "Artificial intelligence mimics human cognition. AI learns from data.",
    ],
    
    '예술': [
        "Music is organized sound. Music expresses emotions.",
        "Painting is the application of color. Painting creates visual art.",
        "Sculpture is three-dimensional art. Sculpture shapes physical forms.",
        "Dance is rhythmic movement. Dance communicates through body.",
        "Poetry is literary art. Poetry uses language aesthetically.",
        "Theater is dramatic performance. Theater tells stories.",
        "Architecture designs buildings. Architecture combines form and function.",
    ],
    
    '사회': [
        "Politics is the art of governance. Politics shapes society.",
        "Economics studies resource allocation. Economics affects prosperity.",
        "Law is a system of rules. Law maintains social order.",
        "Education is the process of learning. Education develops potential.",
        "Culture is shared beliefs and practices. Culture defines identity.",
        "Religion is organized faith. Religion provides meaning.",
        "Ethics studies moral principles. Ethics guides behavior.",
    ],
    
    '인간': [
        "Consciousness is subjective awareness. Consciousness defines experience.",
        "Memory is stored information. Memory shapes identity.",
        "Perception is sensory interpretation. Perception creates reality.",
        "Reasoning is logical thinking. Reasoning solves problems.",
        "Creativity is novel combination. Creativity generates new ideas.",
        "Will is intentional choice. Will drives action.",
        "Identity is sense of self. Identity evolves over time.",
    ],
    
    '관계': [
        "Cooperation is working together. Cooperation achieves shared goals.",
        "Competition is rivalry for resources. Competition drives improvement.",
        "Communication exchanges information. Communication enables understanding.",
        "Conflict is disagreement or struggle. Conflict requires resolution.",
        "Collaboration combines efforts. Collaboration creates synergy.",
        "Leadership guides groups. Leadership inspires action.",
        "Empathy is understanding others. Empathy creates connection.",
    ],
    
    '자연': [
        "Ocean is a large body of saltwater. Ocean covers most of Earth.",
        "Forest is a dense area of trees. Forest produces oxygen.",
        "Mountain is elevated landform. Mountain reaches great heights.",
        "River is flowing water. River shapes landscapes.",
        "Desert is dry barren land. Desert has extreme conditions.",
        "Climate is weather patterns. Climate affects ecosystems.",
        "Season is a period of year. Season brings changes.",
    ],
    
    '추상': [
        "Infinity is endlessness. Infinity transcends limits.",
        "Eternity is infinite time. Eternity has no beginning or end.",
        "Nothing is absence of existence. Nothing is a profound concept.",
        "Everything is all that exists. Everything includes all possibilities.",
        "Possible means can occur. Possible contrasts with impossible.",
        "Necessary means must occur. Necessary is unavoidable.",
        "Contingent depends on conditions. Contingent may or may not happen.",
    ]
}

def generate_sentences(count=1000):
    """다양한 문장 생성"""
    sentences = []
    
    # 모든 주제의 문장 수집
    all_sentences = []
    for topic, texts in TOPICS.items():
        all_sentences.extend(texts)
    
    # count개만큼 반복 (랜덤)
    for _ in range(count):
        sentences.append(random.choice(all_sentences))
    
    return sentences

def main():
    print("\n" + "="*70)
    print("🌍 초대규모 다양한 언어 학습!")
    print("="*70 + "\n")
    
    learning = RapidLearningEngine()
    
    initial_stats = learning.get_learning_stats()
    print(f"초기: {initial_stats['seeds_stored']:,}개 Seeds\n")
    
    # 훨씬 더 큰 규모
    TOTAL_SENTENCES = 5000  # 5000개 문장
    BATCH_SIZE = 100
    
    print(f"목표: {TOTAL_SENTENCES:,}개 문장 학습")
    print(f"배치: {BATCH_SIZE}개씩\n")
    
    sentences = generate_sentences(TOTAL_SENTENCES)
    
    print("학습 시작...\n")
    start_time = time.time()
    
    learned_count = 0
    for i in range(0, len(sentences), BATCH_SIZE):
        batch_start = time.time()
        batch = sentences[i:i+BATCH_SIZE]
        
        for sentence in batch:
            learning.learn_from_text_ultra_fast(sentence)
            learned_count += 1
        
        batch_time = time.time() - batch_start
        
        # 진행상황
        if (i + BATCH_SIZE) % 500 == 0 or (i + BATCH_SIZE) >= len(sentences):
            stats = learning.get_learning_stats()
            elapsed = time.time() - start_time
            
            print(f"진행: {learned_count:,}/{TOTAL_SENTENCES:,} 문장")
            print(f"  Seeds: {stats['seeds_stored']:,}개")
            print(f"  Bloom: {stats['bloomed_nodes']}개")
            print(f"  시간: {elapsed:.1f}초")
            print(f"  속도: {learned_count/elapsed:.0f} 문장/초\n")
    
    # 최종
    final_stats = learning.get_learning_stats()
    total_time = time.time() - start_time
    
    print("="*70)
    print("📊 최종 결과")
    print("="*70)
    print(f"\n학습 문장: {TOTAL_SENTENCES:,}개")
    print(f"총 Seeds: {final_stats['seeds_stored']:,}개")
    print(f"Bloom: {final_stats['bloomed_nodes']}개")
    print(f"에너지: {final_stats['total_energy']:.1f}")
    print(f"총 시간: {total_time:.1f}초")
    print(f"\n속도: {TOTAL_SENTENCES/total_time:.0f} 문장/초")
    print(f"      {final_stats['seeds_stored']/total_time:.0f} 개념/초\n")
    
    # 증가량
    增加 = final_stats['seeds_stored'] - initial_stats['seeds_stored']
    print(f"새로 학습: {增加:,}개 개념")
    
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    main()
