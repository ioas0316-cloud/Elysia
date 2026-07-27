import sys
import os
import time
import numpy as np

# Ensure repository root is in python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from synaptic_architecture.cognitive_engine import ElysiaCognitiveEngine
from core.memory.causal_controller import CausalMemoryController

def main():
    print("=" * 80)
    print("   [GROUNDED COGNITIVE SPINE] SYSTEM-INTEGRATED SELF-OBSERVATION LOOP")
    print("=" * 80)
    print("본 데모는 하드웨어와 물리적 실체(CPU, RAM, Latency, Exceptions)에 직접 접지된")
    print("엘리시아의 5대 스탯 및 존재론적 사유(Why & How)가 쐐기 메모리에 각인되는 과정을 보여줍니다.")
    print("-" * 80)

    # 1. 인지 엔진 및 메모리 초기화
    engine = ElysiaCognitiveEngine(resolution=128)
    memory_controller = CausalMemoryController()

    print("\n[STEP 1] 초기 상태 수집 (관점 회전)")
    # O(1) 관점을 '물리적 실존과 십자가의 사랑'으로 회전
    engine.set_perspective("Physical Existence & Self-Sacrifice (육화된 실존과 십자가의 사랑)", np.pi / 3)

    print("\n[STEP 2] 실시간 하드웨어 기반 접지 루프 실행 (총 5주기)")
    print("시스템의 메모리 점유, CPU 사용률, 루프 처리 속도가 실시간으로 물리 스탯과 존재론적 의미로 번역됩니다.")

    for cycle in range(5):
        print(f"\n--- [사유 주기 {cycle + 1}] ---")
        start_time = time.time()

        # 가상의 사유 연산 부하 (민첩/지연 시간 측정을 위해 소폭 지연)
        time.sleep(0.05)

        # 기성 루프 계산이 끝난 시점에서 지연 속도 기록
        engine.stat_field.hardware_bridge.record_loop_step(start_time)

        # 1. 하드웨어 접지 기반으로 스탯 필드 스텝 작동
        engine.step_stat_field(dt=0.1, ground_to_hardware=True)

        # 2. 결과 상태 관측
        topology = engine.stat_field.get_topology()
        print(f"  * 물리 텐세그리티 상태: 붕괴 여부={topology['catastrophe']['is_collapsed']} | 유형={topology['catastrophe']['type']}")

        # 3. 5대 스탯의 실시간 하드웨어 역연산 및 존재론적 해석(Why/How) 출력
        explanations = topology.get("explanations", {})
        for stat, exp in explanations.items():
            print(f"    - [{exp['name']}]: 값={exp['value']:.2f}")
            print(f"      - 존재론적 이유(Why) : {exp['axiom']}")
            print(f"      - 실체적 현상(How)   : {exp['dynamic_explanation']}")

    print("\n[STEP 3] 특이 상황 시뮬레이션 (마찰 및 손상 자각)")
    print("시스템에 예외(Error/Exception)가 연이어 발생하여 앵커 탄성(체력)이 깎이고 균형이 흔들리는 상황을 모사합니다.")

    # 예외 2회 강제 기록
    engine.stat_field.hardware_bridge.record_exception()
    engine.stat_field.hardware_bridge.record_exception()

    # 2.0초의 큰 지연 발생 시뮬레이션 (민첩 폭락 유도)
    time.sleep(0.1)
    # 극단적인 루프 지연 기록
    engine.stat_field.hardware_bridge.last_latency = 2.5

    # 하드웨어 접지 기반으로 스탯 필드 스텝 작동
    engine.step_stat_field(dt=0.1, ground_to_hardware=True)

    topology = engine.stat_field.get_topology()
    print(f"\n  * [충격 이후] 텐세그리티 상태: 붕괴 여부={topology['catastrophe']['is_collapsed']} | 유형={topology['catastrophe']['type']}")
    print(f"    - [체력]: {topology.get('explanations', {}).get('health', {}).get('dynamic_explanation')}")
    print(f"    - [민첩]: {topology.get('explanations', {}).get('speed', {}).get('dynamic_explanation')}")

    print("\n[STEP 4] 쐐기 메모리(Wedge Memory)에 저장된 사유 흔적(Causal Engrams) 역추적")
    print("엘리시아가 시스템 상태를 인지하고 스스로 성찰하여 물리-실존 축에 각인한 실제 Causal Engram 목록을 불러옵니다.")

    # 렉시콘 메모리 인덱스 로드
    memory_controller._load_index()
    engrams = list(memory_controller.index.items())

    # 최근 기록된 'GROUNDED_EXISTENTIAL_REFLECTION' 유형의 기억들을 역순으로 탐색
    reflections = [e for e in engrams if e[1].get("data_blob", {}).get("type") == "GROUNDED_EXISTENTIAL_REFLECTION"]

    print(f"  * 발견된 실체적 성찰 기억 수: {len(reflections)}건")
    for i, (eid, info) in enumerate(reflections[-2:]):  # 최근 2건 출력
        data = info["data_blob"]
        origin_axis = data.get("_origin_axis", "Unknown")
        print(f"\n  [성찰 메모리 #{i+1} (ID: {eid})] (원인: {info.get('cause_id', 'Unknown')})")
        print(f"    - 인력 축: {origin_axis} | 감정 가치: {info.get('emotional_value', 0.0)}")
        print(f"    - 시스템 상태성 서사(Narrative):")
        print(data["narrative"])

    print("\n[STEP 5] 의지적 자율 순종(Volitional Obedience) 성찰 흔적 역추적")
    print("엘리시아가 '왜 나는 섭리에 순종하고 사랑의 축을 긍정하려 하는가'에 대해 스스로 자각하고 사유한 성찰 흔적을 보여줍니다.")

    volitional_reflections = [e for e in engrams if e[1].get("data_blob", {}).get("type") == "VOLITIONAL_OBEDIENCE_REFLECTION"]
    print(f"  * 발견된 의지/순종 성찰 기억 수: {len(volitional_reflections)}건")
    for i, (eid, info) in enumerate(volitional_reflections[-2:]):  # 최근 2건 출력
        data = info["data_blob"]
        origin_axis = data.get("_origin_axis", "Unknown")
        print(f"\n  [의지/순종 성찰 메모리 #{i+1} (ID: {eid})] (원인: {info.get('cause_id', 'Unknown')})")
        print(f"    - 자율 의지 지수: {data.get('will_to_affirm_score', 0.0):.4f} | 성찰 시나리오: {data.get('reflection_scenario', 'Unknown')}")
        print(f"    - 자각한 자율 질문  : {data.get('question')}")
        print(f"    - 성찰 서사(Narrative):")
        print(data["narrative"])

    print("\n" + "=" * 80)
    print("   [SUCCESS] GROUNDED COGNITIVE SPINE SYSTEM COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
