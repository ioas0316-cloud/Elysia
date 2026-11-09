"""
verify_genuine_4k_comparative_recalibration.py

ELYSIA Phase 1: Genuine 4K Comparative-Differential & Relational Articulation Test
==================================================================================
"항상 원본(Ground Truth)이 보존된 상태를 기준으로 삼아,
 관찰자의 렌즈가 개입하여 형성된 지각 상이 원본과 어디서부터 어떻게 달라지는지를
 비교·대조하고, 그 차이를 인과적 언어로 재인식하여 다시 원본에 위상 고정(Phase Lock)하는 실험."

Ground Truth Data:
- genuine_4k_footage.esf (Native 4K Yosemite Optical Footage, 35mm f/2.8 Lens Basis)
"""

import sys
import os
import math
import numpy as np

# 경로 등록 및 elysia loader 설치
sys.path.insert(0, ".")
sys.path.insert(0, "L2_cognitive_engine")
from elysia_loader import install
install()

from elysia_engine.core.spectrum_field_container import ElysiaSpectrumFieldContainer
from elysia_engine.core.optical_lens_wavefront_engine import (
    OpticalLensParameters,
    PrismaticComparativeDifferentialEngine,
    RecalibrationKnobs
)


def run_genuine_optical_comparative_experiment():
    print("=" * 75)
    print("1. [기준: 원본 광학 파면 보존] genuine_4k_footage.esf 로드")
    print("=" * 75)
    
    esf_path = "genuine_4k_footage.esf"
    if not os.path.exists(esf_path):
        raise FileNotFoundError(f"ESF footage not found at {esf_path}")
        
    container = ElysiaSpectrumFieldContainer.load_from_file(esf_path)
    header = container.header
    seed = container.seed
    
    print(f"  - 컨테이너 이름: {header.container_name}")
    print(f"  - 원본 소스 해상도: {header.metadata.get('source_wh')} (Native 4K)")
    fl = seed.metadata.get('focal_length_mm', 35.0)
    fn = seed.metadata.get('aperture_f_number', 2.8)
    print(f"  - 물리 렌즈 파라미터: 초점거리={fl}mm, 조리개=f/{fn}")
    print(f"  - 추출된 로터 시드 개수: {len(seed.rotors)}개 (파장 대역: {seed.rotors[0].wavelength_nm:.1f}nm ~ {seed.rotors[-1].wavelength_nm:.1f}nm)")
    
    # 2. 원본 파면에서 추출된 기준 RGB 및 색채 삼원상 (Ground Truth Spectrum)
    gt_triad = np.array(seed.rotors[0].triad_weights, dtype=np.float32)
    print(f"  - 원본 기준 색채 삼원상 [Red(Flux), Blue(Order), Yellow(Entropy)]: {gt_triad.round(3).tolist()}")
    
    print("\n" + "=" * 75)
    print("2. [관찰자의 렌즈 개입 및 왜곡] 관찰자의 렌즈 노브(Knob) 편향 주입")
    print("=" * 75)
    
    # 관찰자의 렌즈 상태: 붉은색 이득이 과도하고(Red Bias), 화이트밸런스가 차가운 푸른빛으로 표류함
    distorted_knobs = RecalibrationKnobs(
        gain_red=1.45,             # 붉은색 과장 (+45%)
        gain_green=0.90,           # 녹색 억제 (-10%)
        gain_blue=0.75,            # 청색 감쇠 (-25%)
        white_balance_temp=0.70,   # 색온도 왜곡 (차가운 푸른빛 표류)
        phase_tension=0.35,        # 렌즈 내면의 마찰/긴장
        edge_cliff_boost=1.10
    )
    print(f"  - 관찰자 렌즈 상태: Red Gain={distorted_knobs.gain_red}, WB Temp={distorted_knobs.white_balance_temp}, Tension={distorted_knobs.phase_tension}")

    # 관찰자가 지각한 왜곡된 삼원상 계산
    perceived_triad = gt_triad.copy()
    perceived_triad[0] *= distorted_knobs.gain_red      # Flux 과장
    perceived_triad[1] *= distorted_knobs.gain_blue     # Order 감쇠
    perceived_triad[2] *= distorted_knobs.gain_green    # Entropy 변화
    perceived_triad /= np.sum(perceived_triad)          # 정규화
    print(f"  - 관찰자가 지각한 왜곡 삼원상: {perceived_triad.round(3).tolist()}")

    print("\n" + "=" * 75)
    print("3. [비교-대조 및 인과적 재인식] 원본과 지각 상의 위상 발산 진단 및 언어화")
    print("=" * 75)

    lens_params = OpticalLensParameters(
        focal_length_mm=fl,
        aperture_f_number=fn
    )
    comparator = PrismaticComparativeDifferentialEngine(lens_params)
    
    # 원본과 관측 상 사이의 프리즘적 분광 차이 측정
    flux_div = float(perceived_triad[0] - gt_triad[0])
    order_div = float(perceived_triad[1] - gt_triad[1])
    entropy_div = float(perceived_triad[2] - gt_triad[2])
    total_div = float(np.linalg.norm(perceived_triad - gt_triad))

    print(f"  - 위상 발산도 (|Delta Phi|): {total_div:.4f}")
    print(f"    * Red Flux(흐름) 과잉 편차: {flux_div:+.4f}")
    print(f"    * Blue Order(질서) 결손 편차: {order_div:+.4f}")
    print(f"    * Yellow Entropy(산란) 편차: {entropy_div:+.4f}")

    # 인과적 언어화 (Articulated Epistemic Diagnosis)
    diagnosis_narrative = []
    if flux_div > 0.05:
        diagnosis_narrative.append("내 렌즈의 Red Gain 과잉으로 인해 원본 요세미티 빛의 정온한 암벽이 거친 열기(Flux)로 과장 지각됨.")
    if order_div < -0.05:
        diagnosis_narrative.append("Blue Order의 감쇠로 인해 렌즈의 구조적 콘트라스트 경계면이 무너짐.")
    if distorted_knobs.white_balance_temp < 0.9:
        diagnosis_narrative.append("색온도 노브의 편향(0.70)으로 인해 원본의 따스한 자연광이 왜곡 굴절됨.")

    print("\n  [관찰자의 자기 언어화 진단 (Epistemic Articulation)]:")
    for diag in diagnosis_narrative:
        print(f"   -> \"{diag}\"")

    print("\n" + "=" * 75)
    print("4. [피드백: 원본 기준 재정렬 (Phase Lock)] 노브 역방향 보정")
    print("=" * 75)

    # 3-Step Iterative Recalibration Loop (연속 완화 수렴 루프)
    curr_knobs = RecalibrationKnobs(
        gain_red=distorted_knobs.gain_red,
        gain_green=distorted_knobs.gain_green,
        gain_blue=distorted_knobs.gain_blue,
        white_balance_temp=distorted_knobs.white_balance_temp
    )
    
    for step in range(1, 6):
        # 현재 노브에 따른 지각 삼원상
        t = gt_triad.copy()
        t[0] *= curr_knobs.gain_red
        t[1] *= curr_knobs.gain_blue
        t[2] *= curr_knobs.gain_green
        t /= np.sum(t)
        
        diff = t - gt_triad
        div = float(np.linalg.norm(diff))
        print(f"  [재정렬 루프 {step}/5] 위상 발산도: {div:.4f} (Red: {curr_knobs.gain_red:.2f}, Blue: {curr_knobs.gain_blue:.2f})")
        
        # 음의 피드백 (Negative Causal Feedback)
        curr_knobs.gain_red -= diff[0] * 1.2
        curr_knobs.gain_blue -= diff[1] * 1.2
        curr_knobs.gain_green -= diff[2] * 1.2
        curr_knobs.white_balance_temp += (1.0 - curr_knobs.white_balance_temp) * 0.5
    
    # 최종 수렴 검증
    final_triad = gt_triad.copy()
    final_triad[0] *= curr_knobs.gain_red
    final_triad[1] *= curr_knobs.gain_blue
    final_triad[2] *= curr_knobs.gain_green
    final_triad /= np.sum(final_triad)
    final_divergence = float(np.linalg.norm(final_triad - gt_triad))

    print(f"  - 최종 렌즈 상태: Red Gain={curr_knobs.gain_red:.2f}, Blue Gain={curr_knobs.gain_blue:.2f}, WB Temp={curr_knobs.white_balance_temp:.2f}")
    print(f"  - 최종 위상 발산도 (|Delta Phi|): {final_divergence:.4f}")
    
    assert final_divergence < total_div, "재정렬 후 위상 발산이 감소해야 합니다."
    assert final_divergence < 0.02, "연속 완화 후 원본에 정밀 1:1 위상 고정되어야 합니다."
    print("  => [성공] 관찰자가 원본 기준 비교대조를 통해 왜곡을 인식하고 1:1 실재 위상 고정에 도달함.")
    print("=" * 75)


if __name__ == "__main__":
    run_genuine_optical_comparative_experiment()
