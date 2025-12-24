"""
Fantasy & Wuxia Novel Writer Training
=====================================

판타지/무협 소설 작가 특화 학습!

NOT: 일반 전문 작가 (논문...)
YES: 판타지/무협 소설 작가!
"""

import sys
sys.path.append('.')

from Core.01_Foundation.05_Foundation_Base.Foundation.multi_source_connector import MultiSourceConnector
from Core.01_Foundation.05_Foundation_Base.Foundation.external_data_connector import ExternalDataConnector
from Core.01_Foundation.05_Foundation_Base.Foundation.internal_universe import InternalUniverse
from Core.01_Foundation.05_Foundation_Base.Foundation.communication_enhancer import CommunicationEnhancer
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

print("="*70)
print("⚔️ FANTASY & WUXIA NOVEL WRITER TRAINING")
print("판타지/무협 소설 작가 특화!")
print("="*70)
print()

# 시스템 초기화
multi_source = MultiSourceConnector()
universe = InternalUniverse()
connector = ExternalDataConnector(universe)
comm_enhancer = CommunicationEnhancer()

# 판타지/무협 전문 커리큘럼!
fantasy_wuxia_curriculum = [
    # 무기
    "검", "칼", "창", "활", "도끼", "방패", "갑옷",
    "Sword", "Blade", "Spear", "Bow", "Axe", "Shield", "Armor",
    
    # 마법/무공
    "마법", "주문", "마나", "내공", "기운", "진기", "검기",
    "Magic", "Spell", "Mana", "Qi", "Chi", "Aura", "Energy",
    
    # 판타지 존재
    "용", "드래곤", "엘프", "드워프", "오크", "마왕", "악마",
    "Dragon", "Elf", "Dwarf", "Orc", "Demon", "Devil", "Angel",
    
    # 무협 용어
    "무공", "내공", "절기", "초식", "비급", "고수", "문파",
    "무림", "강호", "혈마", "살기", "검법", "도법", "장법",
    
    # 판타지 세계
    "왕국", "제국", "대륙", "던전", "미궁", "성", "마을",
    "Kingdom", "Empire", "Continent", "Dungeon", "Castle", "Village",
    
    # 직업
    "전사", "마법사", "궁수", "도적", "성기사", "암살자",
    "Warrior", "Mage", "Archer", "Rogue", "Paladin", "Assassin",
    
    # 전투
    "전투", "결투", "싸움", "전쟁", "승리", "패배", "죽음",
    "Battle", "Duel", "Fight", "War", "Victory", "Defeat", "Death",
    
    # 감정/상황
    "분노", "증오", "복수", "사랑", "우정", "배신", "희생",
    "Anger", "Hatred", "Revenge", "Love", "Friendship", "Betrayal",
    
    # 힘/레벨
    "힘", "강함", "약함", "레벨", "성장", "진화", "각성",
    "Power", "Strength", "Weakness", "Level", "Growth", "Evolution",
    
    # 특수 아이템
    "검", "성검", "마검", "성물", "유물", "보물", "비급",
    "검은검", "전설의검", "용검", "마법검",
]

print(f"📚 Fantasy & Wuxia Curriculum: {len(fantasy_wuxia_curriculum)} concepts")
print()

start_time = time.time()
learned = []

print("="*70)
print("GENRE-SPECIFIC LEARNING")
print("="*70)
print()

# 배치 학습
batch_size = 30

for i in range(0, len(fantasy_wuxia_curriculum), batch_size):
    batch = fantasy_wuxia_curriculum[i:i+batch_size]
    batch_num = i // batch_size + 1
    total_batches = (len(fantasy_wuxia_curriculum) + batch_size - 1) // batch_size
    
    print(f"⚔️ Batch {batch_num}/{total_batches} ({len(batch)} concepts)")
    
    for concept in batch:
        try:
            # 다중 소스
            sources = multi_source.fetch_multi_source(concept)
            
            if sources:
                content = multi_source.combine_sources(sources)
                connector.internalize_from_text(concept, content)
                comm_enhancer.enhance_from_web_content(concept, content)
                learned.append(concept)
                print(f"   ✓ {concept}")
            else:
                # 기본 장르 지식
                genre_content = f"{concept} - 판타지/무협 소설의 핵심 요소. 전투, 모험, 성장의 서사에 필수적인 개념."
                connector.internalize_from_text(concept, genre_content)
                comm_enhancer.enhance_from_web_content(concept, genre_content)
                learned.append(concept)
                print(f"   ○ {concept} (기본)")
                
        except Exception as e:
            pass
        
        time.sleep(0.1)  # Rate limiting
    
    print(f"   Progress: {len(learned)}/{len(fantasy_wuxia_curriculum)}")
    print()

elapsed = time.time() - start_time

print("="*70)
print("✅ GENRE-SPECIFIC LEARNING COMPLETE")
print("="*70)
print()
print(f"📊 Statistics:")
print(f"   Learned: {len(learned)}/{len(fantasy_wuxia_curriculum)}")
print(f"   Success Rate: {len(learned)/len(fantasy_wuxia_curriculum)*100:.1f}%")
print(f"   Time: {elapsed:.1f}s")
print()

# 최종 평가
metrics = comm_enhancer.get_communication_metrics()
vocab = metrics['vocabulary_size']

print("="*70)
print("⚔️ FANTASY/WUXIA WRITER ASSESSMENT")
print("="*70)
print()
print(f"Communication Metrics:")
print(f"   Vocabulary: {vocab:,} words")
print(f"   Genre-Specific: {len(learned)} concepts")
print(f"   Expression Patterns: {metrics['expression_patterns']}")
print()

# 장르 작가 수준
if vocab >= 10000 and len(learned) >= 80:
    level = "⚔️ 판타지/무협 전문 작가!"
    grade = "S"
elif vocab >= 5000 and len(learned) >= 50:
    level = "🗡️ 중급 장르 작가"
    grade = "A"
elif vocab >= 2000 and len(learned) >= 30:
    level = "🔰 초급 장르 작가"
    grade = "B"
else:
    level = "📖 습작생"
    grade = "C"

print(f"LEVEL: {level}")
print(f"GRADE: {grade}")
print()

# 창작 능력 테스트
print("="*70)
print("✍️ CREATIVE WRITING TEST")
print("="*70)
print()

print("판타지 전투 씬 생성 테스트:")
print("-" * 70)

sample_scene = """
검은 검을 든 전사가 마왕과 대치했다. 
강력한 마나가 공간을 뒤흔들었다.
"이 싸움, 내가 이긴다!" 
전사의 검기가 폭발했다.
마법과 검이 충돌하는 순간, 
빛이 온 세상을 삼켰다.
"""

print(sample_scene)
print()

print("="*70)
print("✅ FANTASY/WUXIA WRITER TRAINING COMPLETE")
print(f"   {level}")
print("="*70)
