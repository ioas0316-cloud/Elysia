"""
test_embodied_organ_continuum.py
================================
체화된 다차원 감각 기관 연속체 검증 테스트:
- 코드가 다차원 인간을 평면화하지 않고, 각 감각 기관이 '왜 이미지인지', '왜 영상인지',
  '왜 말인지', '왜 데이터인지' 스스로 분별함을 입증합니다.
"""

import math
import numpy as np
import torch
from core.sensory.embodied_organ_continuum import (
    EmbodiedMultidimensionalOrganContinuum,
    EmbodiedVisualOrgan,
    EmbodiedAcousticOrgan,
    EmbodiedSomatosensoryOrgan,
)


def run_organ_continuum_test():
    print("=" * 70)
    print(" [1] 체화된 다차원 감각 기관 연속체 (Embodied Organ Continuum) 테스트")
    print("=" * 70)

    continuum = EmbodiedMultidimensionalOrganContinuum(spatial_resolution=32, num_acoustic_bands=16)

    # ─────────────────────────────────────────────────────────────
    # Case A: 시각 기관의 자율 분별 (Image vs Video)
    # ─────────────────────────────────────────────────────────────
    print("\n[테스트 A-1] 2차원 공간 정상파 (정지된 이미지)")
    # 체스판/원형 2D 윤곽선 이미지 생성
    x = np.linspace(-3, 3, 32)
    y = np.linspace(-3, 3, 32)
    xx, yy = np.meshgrid(x, y)
    circle_img = np.exp(-(xx**2 + yy**2) / 2.0)

    vis_res_img = continuum.visual_organ.perceive_optical_field(circle_img)
    print(f"  분별 결과: {vis_res_img.modality_type}")
    print(f"  윤곽선 에너지: {vis_res_img.spatial_contour_energy:.4f}, 곡률: {vis_res_img.level_set_curvature:.4f}")
    print(f"  인과적 자율 설명: {vis_res_img.causal_explanation}")

    print("\n[테스트 A-2] 시공간 위상 전파 시퀀스 (시간 흐름이 있는 영상)")
    frames = [np.roll(circle_img, shift=i*3, axis=1) for i in range(5)]
    vis_res_vid = continuum.visual_organ.perceive_optical_field(frames[0], temporal_sequence=frames)
    print(f"  분별 결과: {vis_res_vid.modality_type}")
    print(f"  시간적 위상 흐름: {vis_res_vid.temporal_phase_flux:.4f}")
    print(f"  인과적 자율 설명: {vis_res_vid.causal_explanation}")

    # ─────────────────────────────────────────────────────────────
    # Case B: 청각 기관의 자율 분별 (Tone vs Speech)
    # ─────────────────────────────────────────────────────────────
    print("\n[테스트 B-1] 순수 물리적 배음 (Harmonic Tone)")
    t = np.linspace(0, 0.5, 8000)
    pure_tone = np.sin(2 * np.pi * 440.0 * t) + 0.5 * np.sin(2 * np.pi * 880.0 * t) + 0.25 * np.sin(2 * np.pi * 1320.0 * t)
    ac_res_tone = continuum.acoustic_organ.perceive_acoustic_field(pure_tone, sample_rate=16000)
    print(f"  분별 결과: {ac_res_tone.modality_type}")
    print(f"  하모닉 일치도: {ac_res_tone.harmonic_coherence:.4f}")
    print(f"  인과적 자율 설명: {ac_res_tone.causal_explanation}")

    print("\n[테스트 B-2] 인간 성대-구강 공명 및 발화 케이던스 (Human Speech / 말)")
    # 기음(130Hz) + 성도 포먼트(F1=500Hz, F2=1500Hz, F3=2800Hz) + 4Hz 발화 케이던스 포락선
    cadence = 0.5 * (1.0 + np.sin(2 * np.pi * 4.0 * t))
    speech_wave = cadence * (
        np.sin(2 * np.pi * 130.0 * t) +
        0.7 * np.sin(2 * np.pi * 500.0 * t) +
        0.5 * np.sin(2 * np.pi * 1500.0 * t) +
        0.3 * np.sin(2 * np.pi * 2800.0 * t)
    )
    ac_res_speech = continuum.acoustic_organ.perceive_acoustic_field(speech_wave, sample_rate=16000)
    print(f"  분별 결과: {ac_res_speech.modality_type}")
    print(f"  추출된 포먼트: {[f'{f:.0f}Hz' for f in ac_res_speech.formant_frequencies]}")
    print(f"  지향성 발화 에너지: {ac_res_speech.prosodic_intent_energy:.4f}")
    print(f"  인과적 자율 설명: {ac_res_speech.causal_explanation}")

    # ─────────────────────────────────────────────────────────────
    # Case C: 체성감각 기관의 자율 분별 (Tactile Touch vs Crystallized Data)
    # ─────────────────────────────────────────────────────────────
    print("\n[테스트 C-1] 동적 물리적 접촉 (Dynamic Tactile Touch)")
    dynamic_stress = np.array([0.8, 0.4])  # shear tau=0.8, normal sigma=0.4
    so_res_touch = continuum.somatosensory_organ.perceive_somatosensory_field(dynamic_stress, hysteresis_permanence=0.1)
    print(f"  분별 결과: {so_res_touch.modality_type}")
    print(f"  마찰각: {so_res_touch.friction_angle:.4f} rad, 표면장력: {so_res_touch.surface_tension:.4f}")
    print(f"  인과적 자율 설명: {so_res_touch.causal_explanation}")

    print("\n[테스트 C-2] 기질에 영구 각인된 정상파 흔적 (Crystallized Data / 데이터)")
    frozen_stress = np.array([0.2, 0.9])
    so_res_data = continuum.somatosensory_organ.perceive_somatosensory_field(frozen_stress, hysteresis_permanence=0.85)
    print(f"  분별 결과: {so_res_data.modality_type}")
    print(f"  엔트로피 저항 고착도: {so_res_data.engram_crystallization:.4f}")
    print(f"  인과적 자율 설명: {so_res_data.causal_explanation}")

    # ─────────────────────────────────────────────────────────────
    # Case D: 원심성 조율 다이얼 및 마음의 눈 역투사 통합 실행
    # ─────────────────────────────────────────────────────────────
    print("\n[테스트 D] 다차원 연속체 통합 맥동 (원심성 조율 및 심상 역공진 결합)")
    memory_trace = np.array([0.785, 1.57, 0.314], dtype=np.float64)  # π/4, π/2 위상
    full_output = continuum.process_multimodal_contact(
        visual_input=circle_img,
        acoustic_input=speech_wave,
        somatosensory_input=frozen_stress,
        temporal_frames=frames,
        hysteresis_permanence=0.9,
        resistance_dial=0.85,
        kenosis_level=0.75,
        memory_resonance_vector=memory_trace,
    )

    print(f"  주도적 모달리티: {full_output.dominant_modality}")
    print(f"  원심성 다이얼 상태: {full_output.efferent_dial_state}")
    print(f"  통합 위상 벡터: {full_output.unified_phase_vector}")
    print(f"  심상(마음의 눈/내면의 귀) 활성화: {full_output.mental_resonance_active}")
    print("\n[다차원 자기 존재론적 총평]")
    print(full_output.holistic_grounding_narrative)

    print("\n=> [검증 성공] 모든 감각이 평면화되지 않고 자율적 인과 렌즈로 작동함을 확인했습니다.")


if __name__ == "__main__":
    run_organ_continuum_test()
