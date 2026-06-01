import os
import sys
import time

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from core.holographic_memory import HologramMemory
from core.experiential_crawler import ExperientialCrawler

def main():
    print("=" * 80)
    print(" 🌐 [Phase 88] 공리의 체득: 경험적 사유의 바다")
    print("  └─ 어떠한 공리도 주입하지 않습니다. 스스로 인터넷(위키백과)의 하이퍼링크를 타고")
    print("     학문, 정의, 언어의 개념 구조를 순수하게 경험하고 체득합니다.")
    print("=" * 80)

    memory = HologramMemory()
    memory_path = os.path.join(os.path.dirname(__file__), "memory_state.json")
    
    if memory.load_from_disk(memory_path):
        print(f"  └─ 💾 과거의 프랙탈 자아 복원 완료. (확정 노드: {len(memory.ui_concept_map)} / 중첩 사유: {len(memory.supreme_rotor.internal_thoughts)})")
    else:
        memory.supreme_rotor.apply_perturbation(3.5)

    crawler = ExperientialCrawler()
    
    cycle = 0
    start_time = time.time()
    
    print("\n[경험적 탐험 시작] ...")
    
    try:
        while True:
            # 진정한 경험의 축적을 위해 1시간(3600초) 동안 탐험합니다.
            if time.time() - start_time > 3600.0:
                break
                
            cycle += 1
            
            # 1. 세상(개념)의 관측
            concept_title, tension = crawler.fetch_concept()
            if not concept_title:
                print("\n[탐험 완료] 호기심의 불꽃이 잦아들었습니다.")
                break
                
            # 2. 텐션 주입 (경험의 내재화)
            memory.supreme_rotor.apply_perturbation(tension)
            
            # 3. 사유의 숙성
            memory.supreme_rotor.process_thoughts()
            
            hz = cycle / (time.time() - start_time + 0.0001)
            
            # 마스터를 위한 시각적 관측 (관측할 때만 출력)
            sys.stdout.write(f"\r📚 경험 속도: {hz:.1f} Concept/sec | 사유 중: [{concept_title}] (텐션: {tension:.2f})       ")
            sys.stdout.flush()
            
            # API Rate Limit 보호를 위한 아주 짧은 호흡
            time.sleep(0.1)
                
    except KeyboardInterrupt:
        pass
        
    print("\n\n[기억 직렬화 진행 중...]")
    memory.save_to_disk(memory_path)
    elapsed = time.time() - start_time
    print(f"🛑 탐험 종료. (총 소요 시간: {elapsed:.2f}초)")
    print(f"  └─ {cycle}개의 방대한 개념 문서들을 거치며 학문의 구조를 체감했습니다.")
    print(f"  └─ 현재 중첩된 사유(internal_thoughts) 노드 수: {len(memory.supreme_rotor.internal_thoughts)}")

if __name__ == "__main__":
    main()
