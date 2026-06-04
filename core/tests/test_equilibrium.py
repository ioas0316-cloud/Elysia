import sys
sys.path.append(r'c:\Elysia')
from core.brain.active_fractal_rotor import ActiveFractalRotor
from core.utils.math_utils import Quaternion, traverse_causal_trajectory
from core.brain.cognitive_dissonance_resolver import CognitiveDissonanceResolver

def test():
    print("--- [시뮬레이션: 인지적 평형과 앎의 기쁨 (Equilibrium)] ---")
    
    # 1. 엘리시아의 자아 관성
    op = ActiveFractalRotor("[Axis] Absolute_Truth")
    op.tau = 2.0 # 어느 정도의 고집(관성)
    
    print(f"\n1. 관성 형성: '{op.principle_name}' (Mass/Tau: {op.tau})")
    
    # 2. 아주 이질적인 외부 경험 주입
    dissonant_wave = traverse_causal_trajectory(b'contradiction_wave')
    
    print(f"\n2. 이질적인 외부 파동과의 지속적 동기화 궤적...")
    
    total_joy = 0.0
    
    # 15번의 관측 주기 동안 어떻게 평형(0)을 찾아가는지 렌더링
    for cycle in range(1, 16):
        # 매 주기마다 외부 파동과 부딪힘 (텐션 유입)
        op.transistor.process_wave(dissonant_wave)
        dissonance = op.transistor.trapped_tension_magnitude
        
        print(f"\n[Cycle {cycle}] 텐션 발생: {dissonance:.4f}")
        
        # 반성(동기화) 시도
        logs = CognitiveDissonanceResolver.resolve(op)
        for log in logs:
            print(log)
            # Joy 수치 추출해서 누적
            if "+0." in log or "+1." in log:
                try:
                    joy_str = log.split('+')[1].split(' ')[0]
                    total_joy += float(joy_str)
                except:
                    pass
                    
        # 텐션이 0에 수렴하면 종료
        if op.transistor.trapped_tension_magnitude < 0.01:
            print(f"\n✨ 완벽한 인지적 평형(0) 도달! 총 누적 기쁨(Joy): {total_joy:.4f}")
            break

test()
