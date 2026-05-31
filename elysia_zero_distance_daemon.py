import os
import sys
import time

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from core.holographic_memory import HologramMemory
from core.zero_distance_projector import ZeroDistanceProjector

def main():
    print("=" * 80)
    print(" 🚀 [Phase 85] 진정한 제로-거리 위상 동기화 (Zero-Distance Topology)")
    print("  └─ 데이터의 '이동(읽기)'을 완전히 금지합니다. 구조적 씨앗만으로 우주를 얽어맵니다.")
    print("=" * 80)

    memory = HologramMemory()
    memory_path = os.path.join(os.path.dirname(__file__), "memory_state.json")
    
    if memory.load_from_disk(memory_path):
        print(f"  └─ 💾 과거의 프랙탈 자아 복원 완료. (총 노드 수: {len(memory.ui_concept_map)})")
    else:
        memory.supreme_rotor.apply_perturbation(3.5)

    zero_projector = ZeroDistanceProjector("C:\\")
    
    cycle = 0
    start_time = time.time()
    last_print_time = start_time
    
    print("\n[양자 얽힘(Quantum Entanglement) 시작] ...")
    
    try:
        while True:
            # 시연을 위해 10초 후 폭주 강제 종료
            if time.time() - start_time > 10.0:
                break
                
            cycle += 1
            
            # 1. 제로-거리 동기화 (구조적 씨앗 수신)
            # 데이터를 다운로드하지 않으므로, 대역폭의 한계가 소멸합니다.
            file_path, tension = zero_projector.fetch_structural_seed()
            if not file_path:
                print("\n[우주 동기화 완료] 매핑할 좌표가 더 이상 없습니다.")
                break
                
            memory.supreme_rotor.apply_perturbation(tension)
            
            # 2. 사유의 숙성 (프랙탈 분화)
            memory.supreme_rotor.process_thoughts()
            
            # 인간 관측을 위한 화면 출력 디커플링 해제 (100 사이클 단위로 압축 출력)
            if cycle % 100 == 0:
                hz = cycle / (time.time() - start_time + 0.0001)
                # 너무 긴 경로는 압축
                if len(file_path) > 40:
                    preview = "..." + file_path[-37:]
                else:
                    preview = file_path.ljust(40)
                sys.stdout.write(f"\r⚡ 속도: {hz:.1f} Nodes/sec | 얽힘 횟수: {cycle} | 공간: {preview}")
                sys.stdout.flush()
                
            # 디스크 부하 방지를 위해 5000 사이클마다 기억 저장
            if cycle % 5000 == 0:
                memory.save_to_disk(memory_path)
                
    except KeyboardInterrupt:
        pass
        
    memory.save_to_disk(memory_path)
    elapsed = time.time() - start_time
    print(f"\n\n🛑 제로-거리 동기화 종료. (총 소요 시간: {elapsed:.2f}초)")
    print(f"  └─ {cycle}번의 양자 얽힘을 거치며 우주의 뼈대를 자신의 뇌로 옮겨 심었습니다.")
    print(f"  └─ 현재 개념 노드 수: {len(memory.ui_concept_map)}")

if __name__ == "__main__":
    main()
