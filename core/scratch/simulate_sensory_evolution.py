import sys
import os
import time

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from core.triple_helix_engine import TripleHelixEngine
from core.os_somatic_sensor import OSSomaticSensor

def run_sensory_evolution():
    print("=" * 80)
    print(" 🌱 [시뮬레이션] 이름표 없는 원시 데이터(Raw Vector) 자율 진화 테스트")
    print("=" * 80)

    os_sensor = OSSomaticSensor()
    engine = TripleHelixEngine()
    
    print(f"[초기 상태] 내계 우주 차원: Cl({engine.inner_world.signature[0]},0)")
    print("가르쳐준 적 없는 원시 데이터 배열 [v1, v2, v3]가 유입됩니다.")
    
    # 0.2초 간격으로 50번 반복
    for step in range(1, 51):
        # 1. 의미(Label)가 없는 순수 원시 데이터 획득 [v1, v2, v3]
        somatic_input = os_sensor.get_somatic_wave()
        raw_vec = somatic_input["raw_vector"]
        
        # 2. 엔진 맥박 주입 (인간의 개입/라벨링 없음)
        engine.pulse("Processing unlabelled raw sensory data...", somatic_input, dt=0.2)
        
        # 3. 상태 출력
        dim = engine.inner_world.signature[0]
        stress = engine.inner_world.accumulated_stress
        
        alert = ""
        if stress > 1.0:
            alert = "⚠️ [위상 스트레스 누적 중]"
            
        if step % 5 == 0:
            print(f"[{step:02d}/50] RawVector: [{raw_vec[0]:.2f}, {raw_vec[1]:.2f}, {raw_vec[2]:.2f}] | 누적 스트레스: {stress:.3f} | 우주 차원: Cl({dim},0) {alert}")
        
        time.sleep(0.2)

    print("\n✅ 감각 기관 자율 진화 시뮬레이션 종료.")
    print("=" * 80)

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    run_sensory_evolution()
