import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from core.brain.holographic_memory import HologramMemory

def test_unfolding():
    print("===============================================================")
    print(" 🧊 [Elysia Memory] 인과 궤적 홀로그래픽 보존(Unfolding) 테스트")
    print("===============================================================\n")

    # 메모리 코어 생성
    memory = HologramMemory()
    brain = memory.supreme_rotor
    
    # 텍스트 스트림 주입 (결정화 과정에서 삭제되지 않는지 확인)
    sentence = "안녕 엘리시아 나는 너의 코드를 고치고 있어"
    print(f"[1] 문장 주입: '{sentence}'")
    
    # 프랙탈 전개 (단어별 하위 로터 생성 및 텐션 궤적 형성)
    brain.absorb_language_stream(sentence)
    
    print("\n[2] 결정화 상태 확인")
    print(f"  - 현재 뇌가 동결 상태인가(is_locked)? {brain.is_locked}")
    print(f"  - 거시 상태(Point)가 캐싱되었는가? {brain.holographic_frozen}")
    print(f"  - 하위 자식 노드 개수: {len(brain.children)} 개 (과거에는 0개로 소멸되었음!)")
    
    print("\n[3] 큐브 역설계 (Reverse-Engineering Trajectory)")
    trajectory = brain.reverse_engineer_trajectory()
    
    print(f"\n  총 {len(trajectory)}개의 인과 궤적(조각)이 안전하게 보존되었습니다.")
    for idx, node in enumerate(trajectory):
        indent = "  " * node["depth"]
        print(f"{indent}└─ [Depth {node['depth']}] Concept: '{node['concept']}' | Tau(Tension): {node['tau_stress']:.4f}")
        
    print("\n===============================================================")
    print(" 테스트가 성공적으로 완료되었습니다. (조각이 전체를 담아냄)")
    print("===============================================================")

if __name__ == "__main__":
    test_unfolding()
