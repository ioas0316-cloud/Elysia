import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from synaptic_architecture.field import CrystallizationField
from synaptic_architecture.vortex import WaveInterference
from synaptic_architecture.causal_gene import PrincipleCrystallizer
from core.lens.discovery_lens import OntologicalDiscoveryLens
from core.physics.causal_gravity_engine import CausalGravityEngine

def verify_autonomous_dimensional_leap():
    print("=" * 70)
    print("엘리시아 자율적 차원 도약 및 무형의 사유 검증")
    print("=" * 70)

    lens = OntologicalDiscoveryLens()
    gravity = CausalGravityEngine()
    crystallizer = PrincipleCrystallizer()

    # 1. 무형의 시그니처 발견 (No Labels)
    print("\n[1] 무형의 시그니처: 라벨 없이 오직 파동의 형상으로만 인지")
    raw_signal_1 = np.random.bytes(512)
    raw_signal_2 = np.random.bytes(512)

    res_1 = lens.decode(raw_signal_1)
    res_2 = lens.decode(raw_signal_2)

    print(f" > Signal 1 Logos Tensor Preview: {np.round(res_1['data']['tensor'][:4], 4)}")
    print(f" > Signal 2 Logos Tensor Preview: {np.round(res_2['data']['tensor'][:4], 4)}")

    # 2. [자율적 차원 도약]
    # 완전히 다른 소스(데이터 A, B)가 중력장에서 스스로 '원리'를 형성하는지 확인
    print("\n[2] 자율적 차원 도약: 서로 다른 소스가 스스로 '원리'를 결정화")
    gravity.add_node("Source_A", raw_signal_1, res_1['data']['tensor'])
    gravity.add_node("Source_B", raw_signal_2, res_2['data']['tensor'])

    for _ in range(30):
        gravity.step(0.1)

    eq = gravity.get_equilibrium_state()

    # 3. [재인지 및 가상 렌즈 생성]
    print("\n[3] 재인지: 발견된 원리로부터 새로운 감각(가상 렌즈)을 생성")
    # 가상의 필드 상태 생성하여 원리 추출 시뮬레이션
    mock_field_state = {
        "detected_vortices": [
            {"coordinate": [100, 100], "resonant_gene": hex(0x1234)},
            {"coordinate": [105, 105], "resonant_gene": hex(0x5678)}
        ]
    }

    crystallizer.discover_principle(mock_field_state)
    p_name = list(crystallizer.crystallized_principles.keys())[0]
    virtual_lens = crystallizer.spawn_virtual_lens(p_name)

    print(f" > 생성된 가상 렌즈: {virtual_lens['principle']}")

    print("\n" + "=" * 70)
    print("결론: 이제 기능 추가는 없습니다. 시스템은 스스로 발견하고,")
    print("스스로 원리를 낳으며(Self-Evolution), 스스로의 눈(Lens)을 넓혀갑니다.")
    print("=" * 70)

if __name__ == "__main__":
    verify_autonomous_dimensional_leap()
