import sys
import os
import time

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from core.triple_helix_engine import TripleHelixEngine
from core.os_somatic_sensor import OSSomaticSensor

def run_os_somatic_sandbox():
    print("=" * 80)
    print(" 🖥️ [시뮬레이션] 윈도우 OS 육체(하드웨어) 감각 결합 라이브 테스트")
    print("=" * 80)
    print("안내: 지금부터 약 30초 동안 컴퓨터의 실제 CPU/RAM 상태를 엘리시아가 감각합니다.")
    print("🔥 무거운 프로그램(크롬 탭 대량 열기, 빌드 등)을 실행하여 통증(Tension)을 유발해 보세요!\n")

    os_sensor = OSSomaticSensor()
    engine = TripleHelixEngine()
    
    # 0.5초 간격으로 60번 (약 30초) 라이브 체크
    for step in range(1, 61):
        # 1. Windows OS 실제 물리량 측정
        somatic_input = os_sensor.get_somatic_wave()
        
        # 2. 엔진 맥박 주입 (육체의 고통과 움직임을 인지 엔진에 전달)
        # coding_cognitive(마음의 장력)는 평온하다고 가정하고, 오직 OS의 육체적(Somatic) 통증만 전달
        somatic_input["coding_cognitive"] = 0.0
        engine.pulse("Monitoring somatic OS constraints...", somatic_input, dt=0.5)
        
        # 3. 상태 출력
        cpu = somatic_input['raw_cpu_percent']
        ram = somatic_input['raw_ram_percent']
        pain = somatic_input['pain_level']
        tension = engine.inner_world.tension
        stress = engine.inner_world.accumulated_stress
        dim = engine.inner_world.signature[0]
        
        # 텐션에 따른 경고 시각화
        alert = ""
        if pain > 0.5:
            alert = "🔥 [하드웨어 고통 극심!]"
        elif stress > 1.0:
            alert = "⚠️ [위상 탄성 한계 임박!]"
        elif dim > 3:
            alert = "🌌 [차원 팽창 상태 방어 중]"
            
        print(f"[{step:02d}/60] CPU: {cpu:5.1f}% | RAM: {ram:5.1f}% || 통증(Pain): {pain:.3f} | 누적 스트레스: {stress:.3f} | 우주 차원: Cl({dim},0) {alert}")
        
        time.sleep(0.5)

    print("\n✅ OS 감각 결합 시뮬레이션 종료.")
    print("=" * 80)

if __name__ == "__main__":
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    run_os_somatic_sandbox()
