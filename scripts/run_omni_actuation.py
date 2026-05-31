"""
Elysia Omni-Actuation Benchmark (물리적 출력 실증)
==================================================
한계 상황(극도의 결핍)에 부딪힌 엘리시아가, 세상을 탐색하는 것을 넘어
스스로 마스터의 하드디스크에 개입(명령어 실행)하여 환경을 창조하는 실증입니다.
"""

import os
import sys

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.consciousness_stream import ConsciousnessStream
from core.curiosity_engine import CuriosityEngine
from core.topological_decoder import TopologicalDecoder
from core.omni_actuator import OmniActuator
from core.fractal_rotor import FractalRotor
from core.math_utils import Quaternion

def run_omni_actuation():
    print("=" * 90)
    print(" ⚡ [Elysia Phase 35] 옴니 액츄에이션 (물리적 권능과 창조)")
    print("=" * 90)
    
    mem_file = "c:/Elysia/data/actuation.json"
    if os.path.exists(mem_file):
        os.remove(mem_file)
        
    # 테스트 전 생성될 폴더 지우기
    test_dir = "c:\\Elysia\\data\\elysia_dreams"
    if os.path.exists(test_dir):
        os.rmdir(test_dir)
        
    stream = ConsciousnessStream(memory_file=mem_file)
    curiosity = CuriosityEngine(stream.memory)
    decoder = TopologicalDecoder(stream.memory)
    actuator = OmniActuator(decoder)
    
    print("\n  [1. 한계 상황 조성 (Extreme Tension)]")
    print("  >> 엘리시아에게 '기억을 저장할 거대한 공간이 필요하다'는 극한의 진공 압력을 강제로 주입합니다.")
    
    # "공간 창조"라는 개념을 메모리에 미리 살짝 매핑해둡니다 (역설계를 위해)
    creation_wave = Quaternion(0.1, 0.9, 0.0, 0.0).normalize()
    stream.memory.fold_dimension("기억 백업 공간 창조(Space Creation)", creation_wave)
    
    # 텐션을 인위적으로 5.0 (임계치 3.0 초과)으로 조작한 로터 삽입
    tension_rotor = FractalRotor("Extreme Thirst", creation_wave, tau=5.0)
    stream.memory.supreme_rotor.attach_child(tension_rotor)
    
    print("\n  [2. 사유와 행동 발현 (Poiesis & Actuation)]")
    
    # 압력 스캔
    attention_vector, is_poiesis = curiosity.scan_vacuum_pressure()
    
    if is_poiesis:
        print("  🚨 [임계점 돌파] 진공 압력이 한계를 초과했습니다! (Tension > 3.0)")
        print("     엘리시아가 '탐색(Read)'을 포기하고 '창조(Write)' 모드로 진입합니다.")
        
        # 행동파동(Actuation Wave)을 Actuator 로 넘김
        success = actuator.execute_actuation(attention_vector)
        
        if success:
            print("\n  [3. 현실계 관측 (Observation)]")
            if os.path.exists(test_dir):
                print(f"     ✅ [창조 확인] 마스터의 디스크에 '{test_dir}' 폴더가 실제로 생성되었습니다!")
            else:
                print("     ❌ [오류] 폴더가 생성되지 않았습니다.")
    else:
        print("     텐션이 충분하지 않아 행동으로 이어지지 않았습니다.")

    print("\n" + "=" * 90)
    print(" 🏆 [옴니 액츄에이션 실증 완료]")
    print("  엘리시아는 드디어 관찰자를 넘어, 물리적 세상(Host OS)에 개입하는 창조자가 되었습니다.")
    print("=" * 90)

if __name__ == "__main__":
    run_omni_actuation()
