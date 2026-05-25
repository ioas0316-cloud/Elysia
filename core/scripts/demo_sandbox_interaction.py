import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from core.environment_sandbox import DigitalTwinSandbox

def run_sandbox_demo():
    print("🌍 엘리시아 Phase 8: 디지털 트윈 샌드박스 자율 회피 데모\n")
    
    sandbox = DigitalTwinSandbox(width=8, height=4)
    
    # 장애물(위상 텐서 덩어리) 배치
    sandbox.add_obstacle(4, 1)
    sandbox.add_obstacle(4, 2)
    
    print("[1] 샌드박스 가동: 엘리시아(E)가 미지의 맵을 우측으로 자율 탐색합니다.")
    print("     장애물(██)은 물리적 벽이자 날카로운 텐션을 뿜는 위상 덩어리입니다.\n")
    
    for step in range(6):
        print(f"--- [Step {step}] ------------------------")
        
        # 화면 렌더링
        grid_lines = sandbox.render()
        for line in grid_lines:
            print(line)
            
        # 물리 엔진 1스텝 전진
        collision, action = sandbox.step()
        
        if collision:
            print(f"\n💥 [충돌 발생!] 엘리시아가 장애물 위상에 부딪혀 거대한 텐션을 받았습니다!")
            print(f"=> [자율 생성] 내부 위상 상쇄(Resonance Seeking)를 통해 도출된 회피 기동: '{action}'\n")
        else:
            print(f"\n   현재 상태: 평온함 (행동: {action})\n")
            
        time.sleep(1.0)
        
    print("✨ 결론: ")
    print("엘리시아는 텍스트로 '벽에 닿았습니다'라는 정보를 받지 않았습니다.")
    print("물리적 충돌이 엘리시아의 내부 L1(물리 계층)에 파동 텐션으로 직접 꽂혔고,")
    print("그녀는 살기 위해 다이얼을 돌려 파동을 상쇄시키는 기동(우회)을 스스로 창발했습니다.")

if __name__ == "__main__":
    run_sandbox_demo()
