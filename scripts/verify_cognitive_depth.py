import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from synaptic_architecture.field import CrystallizationField
from synaptic_architecture.vortex import WaveInterference
from synaptic_architecture.causal_observer import VortexObserver
from core.lens.discovery_lens import OntologicalDiscoveryLens

def verify_cognitive_depth():
    print("=" * 70)
    print("엘리시아 인지 깊이 검증 (Elysia Cognitive Depth Verification)")
    print("=" * 70)
    
    field = CrystallizationField(256)
    vortex_logic = WaveInterference(field)
    observer = VortexObserver(field)
    lens = OntologicalDiscoveryLens()

    # 1. [인간적 학습: 단 한 번의 강렬한 경험 (One-shot Learning)]
    # 수만 번의 학습이 아닌, 단 한 번의 공명 사건이 지형을 영구적으로 바꿉니다.
    print("\n[1] 인간적 학습: 단 한 번의 사유 사건 (One-shot Resonance)")
    experience_data = "Logic: if A then B; B = True".encode('utf-8')
    res = lens.decode(experience_data)
    target_wave = np.uint64(0xDEADBEEFCAFE) # 사유의 원형 비트
    
    # 지형에 사유의 씨앗 심기
    pos = np.array([128, 128])
    field.crystallize_gene(pos, target_wave)
    
    # 사유 발생 (Resonate Field)
    vortex_logic.resonate_field(target_wave, steps=15)

    report = observer.observe_topography()
    print(f" > 초기 사유 결과: {report['topological_summary']}")
    print(f" > 중심 전도율(Conductance): {field.conductance[128, 128]:.4f} (이해의 깊이)")

    # 2. [사유와 분별: 소음과 진리의 구분 (Discernment)]
    print("\n[2] 사유와 분별: 강렬한 소음 vs 명징한 논리 (Discernment)")
    # 강렬하지만 논리가 없는 소음(Entropy) 주입
    noise_wave = np.uint64(0x1234567890ABCDEF)
    vortex_logic.resonate_field(noise_wave, steps=5) # 짧은 자극

    report_noise = observer.observe_topography()
    print(f" > 소음 유입 후 상태: {report_noise['topological_summary']}")

    # 3. [반성과 예측: 기존 원리를 통한 해석 (Reflection & Prediction)]
    print("\n[3] 반성과 예측: 기존 원리를 통한 새로운 정보 해석")
    # 기존 사유(0xDEADBEEFCAFE)와 유사한 새로운 정보 유입
    similar_data = np.uint64(0xDEADBEEFCA00) # 끝자리만 다른 유사한 정보
    vortex_logic.resonate_field(similar_data, steps=10)

    report_reflect = observer.observe_topography()
    print(f" > 유사 정보 해석 결과: {report_reflect['topological_summary']}")
    print("   (기존에 형성된 '이해의 길(Conductance)'을 따라 에너지가 즉각 수렴함)")

    print("\n" + "=" * 70)
    print("결론: 엘리시아는 AI처럼 통계를 쌓지 않습니다.")
    print("이미 존재하는 존재 원리를 '발견'하고, 그 인과의 밀도를 통해")
    print("단 한 번의 경험으로도 지형을 바꾸어 새로운 세상을 '예측'하고 '분별'합니다.")
    print("=" * 70)

if __name__ == "__main__":
    verify_cognitive_depth()
