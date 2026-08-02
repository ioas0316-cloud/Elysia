"""
Elysia Real-World Multimodal Experience & Concept Mapping Demonstration
=======================================================================
단순화된 무작위 난수 데이터 시뮬레이션을 배격하고,
진짜 세상을 경험하듯 동영상(시각+청각+언어), 실제 이미지, 실제 코드 구조를
인과장에 유입시켜 정의와 개념이 자율적 정상파(Standing Wave)로 매핑되는 과정을 실시간 구동합니다.
"""

import os
import sys
import time
import numpy as np

# UTF-8 콘솔 출력 보장
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# 프로젝트 루트 경로 추가
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.ingestion.real_multimodal_ingester import RealMultimodalPipeline


def main():
    print("=" * 80)
    print(" [ELYSIA REAL-WORLD MULTIMODAL EXPERIENCE & CONCEPT MAPPING ENGINE]")
    print("=" * 80)

    pipeline = RealMultimodalPipeline(field_size=32)

    # -------------------------------------------------------------------------
    # 1. 실제 이미지 자극 경험 (Real Optical Grid Experience)
    # -------------------------------------------------------------------------
    image_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "core", "ingestion", "apple_test.jpg"))
    print(f"\n[1/3] 🖼️ 실제 광학 이미지 자극 유입: {os.path.basename(image_path)}")

    if os.path.exists(image_path):
        res_img = pipeline.ingest_real_image_file(image_path)
        print(f"  ├─ 색채 위상 벡터 (RGB Flux/Order/Entropy): {res_img['perceived_chromatic_vector']}")
        print(f"  ├─ 내재 축 S_abs 굴절률 (Refraction Index): {res_img['refraction_index']:.4f}")
        print(f"  ├─ 잔여 자유 에너지 (Residual Energy): {res_img['residual_free_energy']:.4f}")
        print(f"  ├─ 자발적 정상파 고착 개념 (Emergent Concept): {res_img['emergent_concept']}")
        print(f"  └─ 앎의 사후적 서사 (Narrative): {res_img['narrative']}")

    # -------------------------------------------------------------------------
    # 2. 실제 파이썬 소스 코드 자극 경험 (Computational Friction & AST Experience)
    # -------------------------------------------------------------------------
    code_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "core", "physics", "phase_gravity.py"))
    print(f"\n[2/3] 💻 실제 시스템 코드 구조 유입: {os.path.basename(code_path)}")

    if os.path.exists(code_path):
        res_code = pipeline.ingest_real_source_code(code_path)
        print(f"  ├─ 코드 구조 운동성 (Total Motion/Complexity): {res_code['total_motion']:.4f}")
        print(f"  ├─ 내재 축 S_abs 굴절률 (Refraction Index): {res_code['refraction_index']:.4f}")
        print(f"  ├─ 잔여 자유 에너지 (Residual Energy): {res_code['residual_free_energy']:.4f}")
        print(f"  ├─ 자발적 정상파 고착 개념 (Emergent Concept): {res_code['emergent_concept']}")
        print(f"  └─ 앎의 사후적 서사 (Narrative): {res_code['narrative']}")

    # -------------------------------------------------------------------------
    # 3. 통합 동영상 시공간 연속체 파동 경험 (Unified Spatiotemporal Video Stream)
    # -------------------------------------------------------------------------
    print(f"\n[3/3] 🎬 통합 동영상 시공간 파동 유입 (시각 Frames + 청각 Spectrum + 언어 Subtitles)")

    image_paths = [image_path, image_path, image_path] if os.path.exists(image_path) else []
    audio_freqs = [
        np.array([440.0, 880.0, 1760.0], dtype=np.float32),     # 라(A) 음향 수직 파동
        np.array([523.25, 1046.5, 2093.0], dtype=np.float32),   # 도(C) 음향 수직 파동
        np.array([659.25, 1318.5, 2637.0], dtype=np.float32)    # 미(E) 음향 수직 파동
    ]
    subtitles = [
        "빛의 줄기가 어둠을 베고 시공간 장을 엽니다.",
        "거대한 공명이 온 누리를 울리며 공간을 뒤흔듭니다.",
        "생명의 호흡이 인과장의 결점을 지나 스스로 존재를 선언합니다."
    ]

    res_video = pipeline.ingest_unified_video_sequence(image_paths, audio_freqs, subtitles)
    print(f"  ├─ 수용된 프레임 수: {res_video['packets_count']} 개")
    print(f"  ├─ 통합 청각 에너지 (Audio Energy): {res_video['total_audio_energy']:.4f}")
    print(f"  ├─ 통합 시각 운동량 (Total Motion): {res_video['total_motion']:.4f}")
    print(f"  ├─ 색채 위상 벡터: {res_video['perceived_chromatic_vector']}")
    print(f"  ├─ 내재 축 S_abs 굴절률: {res_video['refraction_index']:.4f}")
    print(f"  ├─ 최종 4D 정상파 좌표: {res_video['standing_coordinate']}")
    print(f"  ├─ 자발적 창발 개념: {res_video['emergent_concept']}")
    print(f"  └─ 앎의 사후적 서사 (Narrative): {res_video['narrative']}")

    print("\n" + "=" * 80)
    print(" 🌟 ELYSIA MULTIMODAL INGESTION & CONCEPT MAPPING COMPLETED CLEANLY.")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
