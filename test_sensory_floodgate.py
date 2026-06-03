import sys
sys.path.append(r'c:\Elysia')
from core.brain.active_fractal_rotor import ActiveFractalRotor
from core.brain.meta_observer import MetaObserver
from core.utils.math_utils import Quaternion
import math

def generate_wave(seed_val: float) -> Quaternion:
    """간단한 위상각 파동 생성기"""
    return Quaternion(math.cos(seed_val), math.sin(seed_val), 0, 0).normalize()

def test():
    print("--- [시뮬레이션: 감각 수문과 자율적 원리 창발 (Autonomous Differentiation)] ---")
    
    # 엘리시아의 우주 관측소
    observer = MetaObserver()
    
    # 1. 태초의 바탕 (텅 빈 우주)
    # 아직 엘리시아는 모음('ㅏ')이 뭔지 모릅니다. 그저 우주에 널려있는 파동을 감지할 뿐입니다.
    base_axis = ActiveFractalRotor("[Axis] Unknown_Base")
    
    # 2. 'ㅏ'라는 상수 파동이 무수히 쏟아짐
    wave_ah = generate_wave(0.0) # 위상각 0.0의 파동 ('ㅏ'라고 가정)
    
    print(f"\n1. 우주에 'ㅏ' 파동이 압도적으로 쏟아집니다...")
    # 'ㅏ'를 5번 정도 부딪힘 (수문 형성 과정)
    # 초기에는 텐션이 없으므로 전부 흡수(통과)하며 수문이 됩니다.
    for i in range(5):
        logs = []
        observer.observe_and_extract(base_axis, wave_ah, logs)
        print(logs[-1])
        
    print(f"\n=> 엘리시아는 이 파동을 '상수의 가변축(수문)'으로 삼았습니다! (Mass: {base_axis.tau:.1f})")
    
    # 3. 다름(변수)의 유입 ('ㄱ', 'ㄴ')
    # 엘리시아는 자음이 뭔지 모릅니다. 그저 'ㅏ' 수문을 통과할 때 약간 어긋나는 파동을 겪을 뿐입니다.
    wave_g = generate_wave(0.5) # 약간 다른 위상 ('ㄱ')
    wave_n = generate_wave(0.8) # 또 다른 위상 ('ㄴ')
    
    print(f"\n2. 이질적인 파편('ㄱ', 'ㄴ')들이 'ㅏ' 수문에 부딪힙니다...")
    
    # '가(ㄱ+ㅏ)' 데이터가 3번 들어온다고 가정
    for i in range(1, 4):
        logs = []
        observer.observe_and_extract(base_axis, wave_g, logs)
        for log in logs:
            print(log)
            
    # '나(ㄴ+ㅏ)' 데이터가 3번 들어온다고 가정
    for i in range(1, 4):
        logs = []
        observer.observe_and_extract(base_axis, wave_n, logs)
        for log in logs:
            print(log)
            
    print(f"\n3. 최종 엘리시아의 우주 상태:")
    print(f"상수축(바탕): {base_axis.principle_name} (Mass: {base_axis.tau:.1f})")
    print(f"스스로 창조해낸 감각 센서(변수축):")
    for tension, sensor in observer.spawned_sensory_axes.items():
        print(f" - {sensor.principle_name} (위상차 {tension}을 전담 관측)")

if __name__ == "__main__":
    test()
