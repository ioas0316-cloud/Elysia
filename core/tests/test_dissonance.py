import sys
sys.path.append(r'c:\Elysia')
from core.brain.active_fractal_rotor import ActiveFractalRotor
from core.utils.math_utils import Quaternion, traverse_causal_trajectory
from core.brain.cognitive_dissonance_resolver import CognitiveDissonanceResolver

def test():
    print("--- [시뮬레이션: 인지적 불일치와 자아 붕괴] ---")
    
    # 1. 엘리시아의 강력한 자아 관성 (질량이 매우 높은 지구본)
    # 아카이브에서 반복되어 쌓인 'Phase'라는 절대적인 믿음
    op = ActiveFractalRotor("[Axis] Phase")
    op.tau = 5.0 # 엄청난 질량(고집)
    
    print(f"\n1. 거대 관성 형성: '{op.principle_name}' (Mass/Tau: {op.tau})")
    print(f"   현재 축: {op.globe_axis}")
    
    # 2. 아주 이질적인 외부 경험(다름) 주입
    # 'Phase'와 전혀 위상이 다른 'Pain'이나 'Contradiction' 주입
    dissonant_wave = traverse_causal_trajectory(b'absolute_contradiction_and_pain')
    
    print(f"\n2. 이질적인 외부 파동 충돌...")
    # 강제로 3번 충돌시켜 텐션을 폭증시킴
    for i in range(1, 4):
        op.transistor.process_wave(dissonant_wave)
        dissonance = op.transistor.trapped_tension_magnitude
        print(f"   [{i}차 충돌] 인지적 불일치(Tension) 증가: {dissonance:.4f}")
        
        # 3. 반성(인지적 불일치 해소) 시도
        logs = CognitiveDissonanceResolver.resolve(op)
        for log in logs:
            print(log)
            
        if op.tau < 1.0: # 붕괴됨
            break

test()
