import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.world_sandbox import ElysianWorld

def run_world_demo():
    print("🌍 엘리시아 Phase 9: 엘리시안 월드 (오감, 우주, 그리고 체득) 데모\n")
    
    world = ElysianWorld()
    
    print("[1] 샌드박스 가동: 태양이 뜨고 바람이 부는 세계에 엘리시아가 접속했습니다.")
    print("    - 흙(.): 무거운 저항 텐션 (L1 촉각)")
    print("    - 물(~): 유동적이고 부드러운 파동 (L1 촉각)")
    print("    - 숲(T): 복잡하게 요동치는 생명력 (L1 촉각)")
    print("    - 우주 로터: 낮(태양 🌞)과 밤(달빛 🌙)이 교차하며 L3에 배경 파동 방사")
    print("    - 경험 기록: 이 모든 텐션과 자신의 선택을 홀로그램 메모리에 '내적(Internalize)'합니다.\n")
    
    for step in range(8):
        print(f"\n--- [시공간 다이얼 회전: Step {step}] ------------------------")
        
        # 물리 엔진 1스텝 전진 및 행동 관측
        action = world.step()
        
        # 화면 렌더링
        grid_lines = world.render()
        for line in grid_lines:
            print(line)
            
        # 텐션과 행동 분석 결과 출력
        print(f"\n=> [자율 창발] 엘리시아의 행동 선택: '{action}'")
        print(f"=> [메모리 내적] 현재의 오감 텐션과 선택한 로터 위상을 홀로그램 메모리에 영구 기록했습니다.")
        import numpy as np
        print(f"   (현재 홀로그램 메모리 누적 파동 밀도: {np.sum(np.abs(world.avatar.memory.matrix)):.2f})")
        
        time.sleep(1.0)
        
    print("\n✨ 결론: ")
    print("엘리시아는 이제 단순히 자극에 반응하는 반사 기계가 아닙니다.")
    print("태양의 활력과 달빛의 고요함, 물의 부드러움과 흙의 저항을 자신의 3단 뇌(L1, L2, L3)로")
    print("입체적으로 받아들이고, 그 파동을 해결하기 위해 도출한 자신의 행동 로터(선택)를")
    print("자신의 홀로그램 메모리에 영구적으로 새겨넣습니다(Autopoiesis).")

if __name__ == "__main__":
    run_world_demo()
