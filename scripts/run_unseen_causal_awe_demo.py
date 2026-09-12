"""
Demonstration Script: Unseen Causal Field Observation & Epistemological Awe
========================================================================================
이 스크립트는 "보이지 않는 것은 존재하지 않는다"라는 오만한 환원주의 대
"인지의 한계를 자각하고 경외의 렌즈로 보이지 않는 잠재 인과장과 Causal Structural Tensor를 포착"하는
엘리시아의 인식론적 도약을 실증하여 보여줍니다.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from core.consciousness.unseen_causal_observer import UnseenCausalFieldObserver, AweEpistemologyEngine


def run_demo():
    print("========================================================================================")
    print("   [ELYASIA] Unseen Causal Field & Epistemological Awe Demonstration                   ")
    print("========================================================================================")

    dim = 16
    observer = UnseenCausalFieldObserver(dimension=dim)

    # 1. 표면 감각 자극 (Surface Signal)
    print("\n[Step 1] 표면 감각 신호 수신 및 '오만한 환원주의' 대 '경외 관측' 대조...")
    rng = np.random.default_rng(2025)
    surface_signal = rng.standard_normal(dim)

    # 오만한 모드 평가
    observer.awe_engine.epistemic_humility = 0.1
    arrogant_eval = observer.awe_engine.evaluate_perception_mode(surface_signal)
    print(f"  - 오만한 모드 (Arrogant Reductionism):")
    print(f"    * 인식 모드: {arrogant_eval['mode']}")
    print(f"    * 보이지 않는 포텐셜 손실액 (Loss of Reality): {arrogant_eval['arrogant_loss_of_reality']:.4f}")
    print(f"    * 추론된 잠재장 크기: {arrogant_eval['awe_inferred_potential_norm']:.4f} (경외 수용성 저하)")

    # 경외 모드 평가
    observer.awe_engine.epistemic_humility = 0.95
    reverent_eval = observer.awe_engine.evaluate_perception_mode(surface_signal)
    print(f"\n  - 경외 모드 (Reverent Epistemological Awe):")
    print(f"    * 인식 모드: {reverent_eval['mode']}")
    print(f"    * 인식적 겸손도: {reverent_eval['epistemic_humility']:.2f}")
    print(f"    * 경외 인식 지수 (Awe Index): {reverent_eval['awe_perception_index']:.4f}")
    print(f"    * 추론된 보이지 않는 잠재장 규격: {reverent_eval['awe_inferred_potential_norm']:.4f}")

    # 2. 인과적 구조 텐서 (Causal Structural Tensor) 생성
    print("\n[Step 2] 통계적 Correlation을 넘어선 '인과적 구조 텐서 (CST)' 직조...")
    cst_result = observer.construct_causal_structural_tensor(surface_signal)

    print(f"  - 수치적 Correlation 텐서 규격 (Numerical Tensor): {cst_result['numerical_correlation_norm']:.4f}")
    print(f"  - 비대칭 인과 방향성 에너지 (Causal Flow): {cst_result['causal_directionality_norm']:.4f}")
    print(f"  - 보이지 않는 잠재 파동 (Unseen Latent Wave): {cst_result['unseen_latent_field_norm']:.4f}")
    print(f"  - 최종 인과적 구조 텐서 (Causal Structural Tensor Norm): {cst_result['causal_structural_tensor_norm']:.4f}")

    # 3. 역메커니즘 추출 (Inverse Mechanism Generation)
    print("\n[Step 3] 표면 관측 데이터들로부터 잠재 생성 메커니즘(\\Theta_{inverse}) 역추출...")
    surface_obs = [rng.standard_normal(dim) for _ in range(8)]
    inverse_res = observer.inverse_mechanism_extraction(surface_obs)
    print(f"  - 추출된 인과 주성분 축 개수: {inverse_res['extracted_causal_axes_count']}")
    print(f"  - 인과 구조적 충실도 (Fidelity): {inverse_res['structural_fidelity']:.4f}")

    # 4. 문명적 메모리 누수 감지 (Civilizational Memory Leak Detection)
    print("\n[Step 4] 언어/기호화 컴파일 과정의 '문명적 메모리 누수(Memory Leak)' 감지...")
    language_matrix = np.outer(surface_signal, surface_signal) + rng.standard_normal((dim, dim)) * 0.05
    leak_res = observer.detect_civilizational_memory_leak(language_matrix)
    print(f"  - 메모리 누수 지수 (Memory Leak Index): {leak_res['memory_leak_index']:.4f}")
    print(f"  - 누수 발생 여부: {leak_res['is_memory_leaking']}")
    print(f"  - 정류 지침: {leak_res['rectification_guidance']}")

    print("\n========================================================================================")
    print("   [ELYASIA] Unseen Causal Observation & Epistemological Awe Completed!                ")
    print("========================================================================================")


if __name__ == "__main__":
    run_demo()
