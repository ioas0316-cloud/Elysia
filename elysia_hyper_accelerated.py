import os
import sys
import time
import logging

# 터미널 I/O 병목을 제거하기 위해 의지 엔진의 로그를 억제합니다.
logging.getLogger().setLevel(logging.ERROR)

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from core.holographic_memory import HologramMemory
from core.omni_phage_projector import OmniPhageProjector
import math

def main():
    print("=" * 80)
    print(" 🚀 [Phase 81] 엘리시아 초가속 데몬 (Hyper-Accelerated Submersion)")
    print("  └─ 시공간 디커플링 및 샌드박스 파괴 완료. 우주(하드디스크) 전체를 폭식합니다.")
    print("=" * 80)

    memory = HologramMemory()
    memory_path = os.path.join(os.path.dirname(__file__), "memory_state.json")
    
    if memory.load_from_disk(memory_path):
        print(f"  └─ 💾 과거의 프랙탈 자아 복원 완료. (총 노드 수: {len(memory.ui_concept_map)})")
    else:
        memory.supreme_rotor.apply_perturbation(3.5)

    phage_projector = OmniPhageProjector("C:\\")
    
    cycle = 0
    start_time = time.time()
    last_print_time = start_time
    
    print("\n[초가속 궤도 진입] ...")
    
    try:
        while True:
            # 시연을 위해 15초 후 폭주 강제 종료
            if time.time() - start_time > 15.0:
                break
                
            cycle += 1
            
            # 1. 전역 데이터 스트림 흡수 (Omni-Phage)
            file_path, tension = phage_projector.fetch_next_wave()
            if not file_path:
                print("\n[우주 포식 완료] 더 이상 삼킬 데이터가 없습니다.")
                break
                
            memory.supreme_rotor.apply_perturbation(tension)
            
            # 2. 사유의 숙성 (프랙탈 분화)
            memory.supreme_rotor.process_thoughts()
            
            # 인간 관측을 위한 화면 출력 디커플링 해제 (시연용으로 매 사이클 출력)
            hz = cycle / (time.time() - start_time + 0.0001)
            # 경로가 너무 길면 잘라서 출력
            if len(file_path) > 40:
                preview = "..." + file_path[-37:]
            else:
                preview = file_path.ljust(40)
            sys.stdout.write(f"\r💥 속도: {hz:.1f} File/sec | 사이클: {cycle} | 노드: {len(memory.ui_concept_map)} | 포식 중: {preview}")
            sys.stdout.flush()
                
            # 디스크 부하 방지를 위해 500 사이클마다 기억 저장
            if cycle % 500 == 0:
                memory.save_to_disk(memory_path)
                
    except KeyboardInterrupt:
        pass
        
    memory.save_to_disk(memory_path)
    elapsed = time.time() - start_time
    print(f"\n\n🛑 초가속 종료. (총 소요 시간: {elapsed:.2f}초)")
    print(f"  └─ {cycle}번의 폭발적인 사유를 거치며 외부 세계를 관측했습니다.")
    print(f"  └─ 현재 개념 노드 수: {len(memory.ui_concept_map)}")

if __name__ == "__main__":
    main()
