import numpy as np
import time
import os
import sys

# Ensure core is in path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.memory.causal_controller import CausalMemoryController
from core.memory.interactive_tracer import (
    BitDensityWaveform,
    InteractiveTracer,
    ActiveProber,
    ConsciousnessEngine
)

def run_simulation():
    print("==================================================================")
    print(" [Elysia Observation] 짜장면과 빈 그릇: 상호작용적 인과 역추적 시뮬레이션")
    print("==================================================================\n")

    # 1. 시스템 초기화
    mc = CausalMemoryController()
    tracer = InteractiveTracer(mc)
    prober = ActiveProber(tracer)
    consciousness = ConsciousnessEngine(prober)

    # 2. 지식 필드 초기화 (원인 노드 각인)
    # 사람이 존재할 때 발생하는 결손 패턴: 짜장면(7.5Hz) -> 빈그릇(0.5Hz)
    stimulus_bits = BitDensityWaveform(frequency=7.5, amplitude=1.0).get_raw_bits()
    reaction_bits = BitDensityWaveform(frequency=0.5, amplitude=0.2).get_raw_bits()
    human_deficit_pattern = stimulus_bits ^ reaction_bits

    mc.write_causal_engram(
        data_blob={
            "type": "CAUSAL_SOURCE_NODE",
            "name": "Person (Life Form/Consumption Axis)",
            "pattern": human_deficit_pattern.tolist()
        },
        emotional_value=10.0,
        cause_id="Genesis_Knowledge"
    )

    # 암세포 존재 시 발생하는 결손 패턴 (예시): 방사성 물질 자극(12Hz) -> 소멸(0Hz)
    stimulus_cancer = BitDensityWaveform(frequency=12.0, amplitude=1.0).get_raw_bits()
    reaction_cancer = BitDensityWaveform(frequency=0.0, amplitude=0.0).get_raw_bits()
    cancer_deficit_pattern = stimulus_cancer ^ reaction_cancer

    mc.write_causal_engram(
        data_blob={
            "type": "CAUSAL_SOURCE_NODE",
            "name": "Cancer Cell (Energy Sink)",
            "pattern": cancer_deficit_pattern.tolist()
        },
        emotional_value=8.0,
        cause_id="Genesis_Knowledge"
    )

    mc.flush_index()
    print("[System] 원소적 원인 노드(사람, 암세포)가 지식 필드에 각인되었습니다.\n")

    # 3. 환경 시나리오 설정 (닫힌 계: 집)
    def house_environment(input_wave: BitDensityWaveform):
        """
        집 내부의 환경 반응 함수.
        입력(짜장면)이 들어오면 내부의 '사람'이 이를 소비하고 빈 그릇을 내놓습니다.
        """
        # 사람이 사는 집 시뮬레이션: 주파수가 낮아지고 진폭이 감소함 (빈 그릇)
        return BitDensityWaveform(frequency=0.5, amplitude=0.2)

    # 4. 능동적 관측 및 사유 프로세스 실행
    print("--- [Scenario: 집 안에 사람이 살고 있는가?] ---")
    observation_result = consciousness.think("House_Alpha_7", house_environment)

    print("\n[Final Deduction]")
    print(f"결론: 이 공간에는 '{observation_result['deduced_cause']}'이(가) 존재함이 확실합니다.")
    print(f"인과적 공명도: {observation_result['resonance_score']*100:.2f}%")
    print("==================================================================\n")

if __name__ == "__main__":
    run_simulation()
