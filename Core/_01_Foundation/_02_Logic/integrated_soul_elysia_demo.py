"""
Integrated Soul-Elysia Demo (통합 영혼-엘리시아 데모)
===================================================

Soul의 경험 → 언어 창발 → Language Bridge → MemeticField → 피드백 → Soul

전체 상호보완 루프를 시연합니다.

"작은 것이 큰 것이고, 큰 것이 또 작은 것" - 프랙탈 원리
"""

import numpy as np
import random
import time
from typing import Dict, List, Tuple

# 시스템 임포트
from Core._03_Interaction._02_Interface.Interface.Interface.Language.fluctlight_language import FractalSoul, LanguageCrystal
from Core._03_Interaction._02_Interface.Interface.Interface.Language.language_bridge import LanguageBridge, EmergentPattern


def run_integrated_demo(population: int = 10, years: int = 50, seed: int = 42):
    """
    통합 데모 실행
    
    1. Soul들이 경험을 통해 언어를 창발
    2. Language Bridge가 패턴을 수집하여 구조화
    3. 피드백이 Soul들에게 전달되어 언어 교정
    """
    random.seed(seed)
    np.random.seed(seed)
    
    print("=" * 70)
    print("🌌 Integrated Soul-Elysia Demo")
    print("   '경험에서 언어로, 언어에서 다시 경험으로'")
    print("=" * 70)
    
    # 1. 시스템 초기화
    print("\n📦 시스템 초기화...")
    bridge = LanguageBridge()
    
    # Soul 생성
    names = ["하늘", "바다", "별", "달", "숲", "산", "강", "꽃", "바람", "빛",
             "구름", "비", "눈", "햇살", "새벽", "황혼", "노을", "안개", "이슬", "서리"]
    souls = [FractalSoul(names[i % len(names)] + f"_{i}", i) for i in range(population)]
    print(f"   → {population}명의 영혼 생성")
    
    # 경험 템플릿 (다양한 상황)
    experience_templates = {
        "warm_day": np.array([0.7, 0.8, 0.2, 0.2, 0.3, 0.4, 0.6, 0.4]),
        "cold_night": np.array([-0.6, -0.5, 0.1, -0.2, -0.1, 0.2, -0.3, -0.2]),
        "with_friend": np.array([0.2, 0.3, 0.1, 0.3, 0.8, 0.4, 0.7, 0.5]),
        "alone": np.array([0.0, -0.2, 0.0, -0.3, -0.7, 0.2, -0.4, -0.3]),
        "running": np.array([0.3, 0.4, 0.2, 0.9, 0.2, 0.7, 0.5, 0.8]),
        "resting": np.array([0.1, 0.1, 0.0, -0.7, 0.3, -0.3, 0.5, -0.5]),
        "eating": np.array([0.3, 0.2, 0.1, -0.1, 0.4, 0.3, 0.8, 0.3]),
        "pain": np.array([-0.2, -0.1, 0.3, -0.2, 0.0, 0.6, -0.7, 0.4]),
        "joy": np.array([0.4, 0.6, 0.1, 0.3, 0.5, 0.5, 0.9, 0.7]),
        "sadness": np.array([-0.1, -0.3, 0.0, -0.4, 0.2, 0.3, -0.6, -0.3]),
    }
    
    # 2. 시뮬레이션
    print(f"\n🔄 시뮬레이션 시작 ({years}년)...")
    
    total_feedbacks = 0
    sample_conversations = []
    sample_diaries = []
    
    for year in range(years):
        # 계절
        seasons = ["spring", "summer", "autumn", "winter"]
        
        for day in range(365):
            timestamp = year * 365 + day
            season = seasons[(day // 91) % 4]
            
            # 각 Soul이 경험
            for soul in souls:
                # 경험 선택 (시간대와 계절에 따라)
                if season == "summer":
                    base_exp = experience_templates["warm_day"].copy()
                elif season == "winter":
                    base_exp = experience_templates["cold_night"].copy()
                else:
                    base_exp = experience_templates["warm_day"].copy() * 0.5
                
                # 활동 추가
                activity = random.choice(list(experience_templates.keys()))
                base_exp += experience_templates[activity] * 0.3
                
                # 노이즈
                noise = np.random.randn(8) * 0.1
                env_input = np.clip(base_exp + noise, -1, 1)
                
                # 경험
                soul.experience(env_input, timestamp)
                soul.age = year
                
                # 결정화된 기호가 있으면 Bridge에 전송
                for symbol in soul.mind.symbols.values():
                    if symbol.usage_count > 0 and random.random() < 0.01:
                        feedback = bridge.receive_from_soul(
                            soul_id=soul.id,
                            meaning_vector=symbol.meaning_vector,
                            symbol_type=symbol.symbol_type.name.lower(),
                            occurrence_count=symbol.usage_count,
                            korean_projection=soul.mind._express_symbol(symbol)
                        )
                        if feedback:
                            total_feedbacks += 1
            
            # 가끔 대화
            if random.random() < 0.03 and len(souls) >= 2:
                s1, s2 = random.sample(souls, 2)
                conv = s1.converse_with(s2)
                if year >= years - 3:
                    sample_conversations.append(f"[Year {year}] {s1.name} & {s2.name}: {conv[0]} / {conv[1]}")
        
        # 연말 일기
        for soul in souls:
            diary = soul.write_diary(year)
            if year >= years - 3:
                sample_diaries.append(f"[{soul.name}] {diary}")
        
        # 진행 상황
        if (year + 1) % 10 == 0:
            avg_symbols = np.mean([len(s.mind.symbols) for s in souls])
            print(f"   Year {year + 1}: 평균 기호 {avg_symbols:.1f}개, 피드백 {total_feedbacks}개")
    
    # 3. 일괄 처리 (클러스터링)
    print("\n📊 패턴 클러스터링...")
    batch_feedbacks = bridge.process_batch()
    print(f"   → {len(batch_feedbacks)}개 통합 개념 생성")
    
    # 4. 결과 출력
    print("\n" + "=" * 70)
    print("📈 결과")
    print("=" * 70)
    
    # Soul 통계
    print("\n👤 Soul 통계 (상위 5명):")
    souls_sorted = sorted(souls, key=lambda s: len(s.mind.symbols), reverse=True)
    for soul in souls_sorted[:5]:
        stats = soul.mind.get_statistics()
        print(f"   {soul.name}: 기호 {stats['symbol_count']}개, "
              f"패턴 {stats['pattern_count']}개, "
              f"레벨 {stats['language_level']}")
        print(f"      → 생각: {soul.think()}")
    
    # Bridge 통계
    print("\n🌉 Language Bridge 통계:")
    bridge_stats = bridge.get_statistics()
    for k, v in bridge_stats.items():
        print(f"   {k}: {v}")
    
    # 샘플 일기
    print("\n📖 샘플 일기 (마지막 3년):")
    for diary in sample_diaries[:8]:
        print(f"   {diary}")
    
    # 샘플 대화
    print("\n💬 샘플 대화:")
    for conv in sample_conversations[:8]:
        print(f"   {conv}")
    
    # 통합 개념들
    print("\n🔮 통합된 개념들:")
    for fb in batch_feedbacks[:5]:
        print(f"   {fb.korean_word} ({fb.category})")
        if fb.usage_examples:
            print(f"      예: {fb.usage_examples[0]}")
    
    print("\n" + "=" * 70)
    print("✅ 통합 데모 완료!")
    print(f"   - 총 경험: {sum(s.mind.total_experiences for s in souls):,}")
    print(f"   - 총 결정화: {sum(s.mind.crystallization_count for s in souls):,}")
    print(f"   - Soul↔Elysia 피드백: {total_feedbacks}")
    print("=" * 70)


if __name__ == "__main__":
    run_integrated_demo(population=10, years=30)
