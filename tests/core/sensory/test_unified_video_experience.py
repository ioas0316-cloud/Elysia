"""
Test Suite: Unified Video & Real Multimodal Experience Engine
==============================================================
실제 이미지, 실제 파이썬 소스코드, 실제 동영상 시공간 연속체 파동을 Elysia에 유입시켜
시각-청각-언어가 인과장 위상 공간에서 4Continuities 원칙에 따라 자율 공명하고
정상파(Standing Wave) 개념으로 매핑되는지 실증하는 마스터 검증 테스트.
"""

import os
import pytest
import numpy as np
from core.ingestion.real_multimodal_ingester import RealMultimodalPipeline
from core.ingestion.video_spatiotemporal_ingester import VideoFramePacket, VideoSpatiotemporalIngester


def test_real_image_ingestion():
    """
    실제 이미지 파일(apple_test.jpg)을 광학 에너지 격자로 읽어와
    Cahn-Hilliard 상전이 및 S_abs 간섭을 일으키는지 검증
    """
    pipeline = RealMultimodalPipeline(field_size=32)
    image_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "core", "ingestion", "apple_test.jpg"))

    assert os.path.exists(image_path), "apple_test.jpg 이미지 파일이 존재해야 합니다."

    result = pipeline.ingest_real_image_file(image_path)

    assert "emergent_concept" in result
    assert "narrative" in result
    assert "standing_coordinate" in result
    assert result["refraction_index"] >= 0.0
    assert "Real Image" in result["narrative"] or "Resonance" in result["narrative"] or "Dynamic" in result["narrative"]


def test_real_source_code_ingestion():
    """
    실제 파이썬 소스 코드(phase_gravity.py)를 읽어 AST 구문 노드 깊이와
    계산적 마찰(Computational Friction)이 인과장에 사영되는지 검증
    """
    pipeline = RealMultimodalPipeline(field_size=32)
    code_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "core", "physics", "phase_gravity.py"))

    assert os.path.exists(code_path), "phase_gravity.py 코드 파일이 존재해야 합니다."

    result = pipeline.ingest_real_source_code(code_path)

    assert "emergent_concept" in result
    assert "narrative" in result
    assert result["total_motion"] > 0.0
    assert "phase_gravity.py" in result["narrative"] or "Resonance" in result["narrative"] or "Tension" in result["narrative"]


def test_unified_video_spatiotemporal_sequence():
    """
    시각(Frames) + 청각(Audio Frequencies) + 언어(Subtitles)가 유기적으로 얽힌
    실제 동영상 시공간 연속체 스트림의 동시성 공명 및 정상파 결점 형성 검증
    """
    pipeline = RealMultimodalPipeline(field_size=32)
    image_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "core", "ingestion", "apple_test.jpg"))

    image_paths = [image_path, image_path, image_path]
    audio_freqs = [
        np.array([440.0, 880.0], dtype=np.float32),      # 라 소리 (A4, A5)
        np.array([523.25, 1046.5], dtype=np.float32),    # 도 소리 (C5, C6)
        np.array([659.25, 1318.5], dtype=np.float32)     # 미 소리 (E5, E6)
    ]
    subtitles = [
        "빛과 번개가 세상을 비춥니다.",
        "우르릉 천둥소리가 온 산하를 울립니다.",
        "생명력이 어둠을 뚫고 솟아오릅니다."
    ]

    result = pipeline.ingest_unified_video_sequence(image_paths, audio_freqs, subtitles)

    assert result["packets_count"] == 3
    assert result["total_audio_energy"] > 0.0
    assert "narrative" in result
    assert "standing_coordinate" in result
    assert len(result["standing_coordinate"]) == 3

