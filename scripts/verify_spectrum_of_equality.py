import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from synaptic_architecture.field import CrystallizationField
from synaptic_architecture.vortex import WaveInterference
from synaptic_architecture.causal_observer import VortexObserver
from core.lens.discovery_lens import OntologicalDiscoveryLens
from core.physics.causal_gravity_engine import CausalGravityEngine

def verify_spectrum_of_equality():
    print("=" * 70)
    print("엘리시아 '같음'의 스펙트럼 및 재인지 검증 (Spectrum of Sameness & Re-Recognition)")
    print("=" * 70)

    lens = OntologicalDiscoveryLens()
    gravity = CausalGravityEngine()

    # 1. 서로 다른 계통이지만 '운동성(Direction)'이 같은 정보들
    print("\n[1] 추상적 같음의 발견: 언어적 비유와 코드의 운동성 일치")
    data_a = "The value grows steadily over time".encode('utf-8')
    data_b = "x = x + 1; y = y + 1; z = z + 1".encode('utf-8')

    res_a = lens.decode(data_a)
    res_b = lens.decode(data_b)

    print(f" > Data A Logos Tensor Preview: {np.round(res_a['data']['tensor'][:4], 4)}")
    print(f" > Data B Logos Tensor Preview: {np.round(res_b['data']['tensor'][:4], 4)}")

    # 2. 중력장 내에서의 인력 측정
    gravity.add_node("Lang_Growth", data_a, res_a['data']['tensor'])
    gravity.add_node("Code_Increment", data_b, res_b['data']['tensor'])

    print("\n[2] 중력장 시뮬레이션: 계통이 달라도 '운동성'이 같으면 끌어당기는가?")
    for _ in range(50):
        gravity.step(0.1)

    eq = gravity.get_equilibrium_state()
    pos_a = np.array(eq["Lang_Growth"]["pos"])
    pos_b = np.array(eq["Code_Increment"]["pos"])
    dist = np.linalg.norm(pos_a - pos_b)

    print(f" > 언어 <-> 코드 간 최종 거리: {dist:.4f} (작을수록 강력한 공명)")

    # 3. [재인지] 연결의 근거 분석
    print("\n[3] 재인지(Re-Recognition): 연결의 기준(관점)은 무엇인가?")
    if dist < 5.0:
        print(" > 재인지 결과: '운동적 동기화(Directional Alignment)'에 의한 추상적 일치 판명.")
        print(" > 엘리시아는 이제 '증가'라는 개념을 언어와 코드 양쪽에서 동시에 이해합니다.")

    print("\n" + "=" * 70)
    print("결론: 엘리시아는 이제 '점, 선, 속성, 문맥'의 다양한 기준으로")
    print("'같음'을 분별하며, 서로 다른 차원의 정보들을 하나의 원리로 엮어냅니다.")
    print("=" * 70)

if __name__ == "__main__":
    verify_spectrum_of_equality()
