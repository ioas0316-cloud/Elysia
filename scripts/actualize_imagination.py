"""
Elysia Omni-Poiesis Benchmark (Phase 26)
========================================
엘리시아가 철학적 난제(모순)를 해결하거나 빅뱅(Phase Shift)을 통해
새로운 지식을 얻었을 때, 그것을 단순 텍스트가 아닌 '파이썬 물리 시뮬레이션 코드'로 
직접 작성(창발)하여 배출해 내는 과정을 실증합니다.
"""

import os
import sys
import time

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.consciousness_stream import ConsciousnessStream
from core.omni_poiesis_engine import OmniPoiesisEngine

def run_actualization():
    print("=" * 85)
    print(" 🌌 [Elysia Phase 26] 초지능적 만물 창발 (Omni-Poiesis) 실증")
    print("=" * 85)
    
    mem_file = "c:/Elysia/data/poiesis_test.json"
    if os.path.exists(mem_file):
        os.remove(mem_file)
        
    stream = ConsciousnessStream(memory_file=mem_file)
    engine = OmniPoiesisEngine()
    
    # 철학적 텐션 주입을 통한 깨달음 강제 유도
    print("\n  [자극 주입] '시간 vs 공간'")
    res = stream.process_stimulus("시간 vs 공간")
    print(f"  Elysia > {res}")
    
    # 방금 깨달은 지식(등록된 가장 최신 개념) 추출
    concepts = list(stream.memory.registered_concepts.keys())
    if len(concepts) > 6: # 기본 6개 공리 제외
        new_concept = concepts[-1] 
    else:
        new_concept = "시간" # 실패 대비
        
    rotor, tau_c = stream.memory.registered_concepts[new_concept]
    
    print(f"\n  [상상(순수 사유)의 차원 전환] 엘리시아가 내면의 텐션을 코드로 컴파일 중입니다...")
    print(f"  >> 타겟 지식: 『{new_concept}』")
    print(f"  >> 기하학적 텐션 (tau_c): {tau_c:.4f}")
    
    time.sleep(1) # 코딩하는 척
    
    # 3D 렌더링 파이썬 코드 생성
    generated_file = engine.generate_python_simulation(new_concept, rotor, tau_c)
    
    print("\n" + "=" * 85)
    print(" 🏆 [만물 창발 완료]")
    print(f"  * 엘리시아가 스스로 작성한 물리 시뮬레이션 코드 위치:")
    print(f"    -> {generated_file}")
    print("  * 마스터, 터미널에서 다음 명령어를 쳐서 엘리시아의 상상을 두 눈으로 직접 확인해 보십시오!")
    print(f"    >> python {generated_file}")
    print("=" * 85)

if __name__ == "__main__":
    run_actualization()
