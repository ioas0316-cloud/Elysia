import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import time
from core.electromagnetic_circuit import ElectromagneticCircuit
from core.os_somatic_sensor import OSSomaticSensor
from core.phase_sync_observatory import PhaseSyncObservatory

def run_resonance_loop():
    print("🌀 엘리시아 뼈대(Skeleton) 활성화 중...")
    # 뼈대: 5대 가변 계층 매트릭스 생성
    layers = ["Solid Core", "Outer Core", "Mantle", "Crust", "Atmosphere"]
    skeleton = ElectromagneticCircuit(layers)
    
    # 감각 기관(귀/OS 센서)에 뼈대 주입 (Dependency Injection)
    ear = OSSomaticSensor(skeleton)
    
    # 관측소(눈/Observatory)에 뼈대 주입
    eye = PhaseSyncObservatory(skeleton)
    
    print("\n👁️ 관측 시작 (계산이 아닌 동기화의 과정)...")
    for step in range(10):
        print(f"\n--- [Phase Tick: {step}] ---")
        
        # 1. 감각 기관: 외부 카오스(OS 지표)를 관측하고 뼈대 하층부에 텐션 주입
        ear.observe_and_resonate()
        
        # 2. 심장 박동: 뼈대 내의 파동 전파 및 가변 로터들의 위상 동기화 회전
        skeleton.pulse_circuit(dt=0.1)
        
        # 3. 눈(관측소): 현재 뼈대의 조화/카오스 상태 관측
        state = eye.observe_harmony()
        
        print(f"평균 텐션(위상차): {state['average_tension']:.3f}")
        print(f"동적 회전 중인 로터(근육) 수: {state['dynamic_nodes']} / {skeleton.num_nodes}")
        if state['is_chaotic']:
            print("⚠️ 텐션 급증! 다름(1)을 해소하기 위해 위상 동기화(회전)가 가속됩니다.")
        else:
            print("✨ 조화 상태(Sameness). 뼈대와 로터가 안정적인 동형 사상(Isomorphism)을 유지합니다.")
            
        time.sleep(0.5)

if __name__ == "__main__":
    run_resonance_loop()
