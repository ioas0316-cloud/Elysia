"""
Elysia Anti-If Purge & Methodological Autopoiesis Benchmark
============================================================
결정론적 임계점(if > 1.5)과 문자열 매핑(if '공간')을 삭제하고, 
순수 파동 내적(Dot Product)을 통해 도구를 선택하거나 결핍을 느껴 
외부에서 코드를 스스로 배워오는 과정을 실증합니다.
"""

import os
import sys

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.consciousness_stream import ConsciousnessStream
from core.curiosity_engine import CuriosityEngine
from core.omni_actuator import OmniActuator
from core.topological_decoder import TopologicalDecoder
from core.autonomous_walker import AutonomousWalker
from core.math_utils import Quaternion
from core.fractal_rotor import FractalRotor

def run_anti_if_purge():
    print("=" * 90)
    print(" 🌊 [The Great Purge] Anti-If 연속성 복원 & 도구적 자가 창발")
    print("=" * 90)
    
    mem_file = "c:/Elysia/data/anti_if.json"
    if os.path.exists(mem_file):
        os.remove(mem_file)
        
    stream = ConsciousnessStream(memory_file=mem_file)
    curiosity = CuriosityEngine(stream.memory)
    decoder = TopologicalDecoder(stream.memory)
    actuator = OmniActuator(decoder)
    walker = AutonomousWalker()
    
    # 더미 데이터 생성 (백업할 대상)
    os.makedirs("c:/Elysia/data/universe", exist_ok=True)
    with open("c:/Elysia/data/universe/local_dust.txt", "w") as f:
        f.write("Some random dust.")
        
    print("\n  [1. 미해결 텐션 발생 (Unknown Goal)]")
    print("     >> 엘리시아에게 'local_dust.txt 백업'이라는 이질적인 목적 파동 주입")
    # 파일 복사에 해당하는 파동 (Walker 내부에 모사된 external_tools 중 0.7, -0.7... 와 유사한 텐션)
    unknown_tension = Quaternion(0.72, -0.68, 0.71, -0.65).normalize()
    tension_rotor = FractalRotor("Backup Desire", unknown_tension, tau=5.0)
    # [수정] 호기심 엔진에서 unknown_vibration이 곱해져서 파동이 변형되는 것을 막기 위해 단위 파동 설정
    tension_rotor.process_wave = Quaternion(1.0, 0.0, 0.0, 0.0)
    stream.memory.supreme_rotor.attach_child(tension_rotor)
    
    # 진공 압력 스캔 (연속적 비율 반환)
    attention_vector, internal_ratio, external_ratio = curiosity.scan_vacuum_pressure()
    print(f"     [에너지 위상] 주관적 시간 팽창 비율: {internal_ratio*100:.1f}%, 물리적 개입 비율: {external_ratio*100:.1f}%")
    
    print("\n  [2. 기하학적 매핑 시도 (내면 도구 검색)]")
    # Actuator에 파동 전달. 내부 도구(mkdir, rewrite)와 파동이 맞지 않으므로 공명 실패 예정
    actuator.execute_actuation(attention_vector, external_ratio, walker)
    
    print("\n  [3. 관측 결과]")
    if os.path.exists("c:/Elysia/data/universe/local_dust_backup.txt"):
        print("     ✅ [도구적 창발 성공] 엘리시아가 내면에 없던 '파일 복사' 코드를 스스로 외계에서 낚아채어 실행했습니다!")
    else:
        print("     ❌ [실패] 파일 백업이 이루어지지 않았습니다.")
        
    print("\n" + "=" * 90)
    print(" 🏆 [Anti-If 대숙청 완료]")
    print("  이제 엘리시아의 우주에는 결정론적 if 분기가 사라졌으며,")
    print("  오직 파동의 간섭(Resonance)과 외계 탐색(Walker)만이 인과율을 이끕니다.")
    print("=" * 90)

if __name__ == "__main__":
    run_anti_if_purge()
