"""
Unit Tests: Binary Container Stripping & Causal World Surgery
=============================================================
컨테이너 박리 100% 무손실 복원, do-수술 및 비트-롯 자가 치유 단위 테스트.
"""

import pytest
import zlib
from core.topology.binary_container_stripper import BinaryContainerStripper
from core.topology.causal_world_surgery_engine import CausalWorldSurgeryEngine


def test_container_stripping_and_100pct_roundtrip():
    stripper = BinaryContainerStripper()
    original_payload = b"ELYSIA_UNIVERSE_INVARIANT_SEED_2026"
    raw_container = stripper.create_sample_container(original_payload)

    # 1. 컨테이너 박리 및 Stem 분리
    graph = stripper.strip_and_parse(raw_container)

    assert len(graph.nodes) == len(original_payload)
    assert graph.context.medium_type == "elys_container"
    assert "magic" in graph.context.environmental_constraints

    # 2. 재합성 (100% 비트 동일성 검증)
    reconstructed = stripper.resynthesize(graph)
    assert reconstructed == raw_container
    assert len(reconstructed) == len(raw_container)


def test_do_operator_surgery():
    stripper = BinaryContainerStripper()
    surgery_engine = CausalWorldSurgeryEngine()

    raw_container = stripper.create_sample_container(b"HELLO_WORLD")
    graph = stripper.strip_and_parse(raw_container)

    # BYTE_0 ('H': 0x48)를 'J' (0x4A)로 do-수술
    mutated_graph = surgery_engine.apply_do_surgery(graph, "BYTE_0", ord("J"))

    assert mutated_graph.nodes["BYTE_0"].payload["val"] == ord("J")
    assert mutated_graph.nodes["BYTE_0"].payload["surgically_modified"]

    # 재합성 후 바이너리가 유효한 새 CRC와 'J' 바이트를 포함하는지 확인
    resynthesized = stripper.resynthesize(mutated_graph)
    assert resynthesized[16:17] == b"J"
    # 새 CRC 일치 확인
    new_crc = zlib.crc32(b"JELLO_WORLD")
    stored_crc = int.from_bytes(resynthesized[12:16], byteorder="big")
    assert stored_crc == new_crc


def test_bit_rot_self_healing():
    stripper = BinaryContainerStripper()
    surgery_engine = CausalWorldSurgeryEngine()

    payload = bytes([10, 20, 30, 40, 50])
    raw_container = stripper.create_sample_container(payload)
    graph = stripper.strip_and_parse(raw_container)

    # 정상 합(Parity Invariant) 보존
    invariant_parity_sum = sum(payload) # 150

    # BYTE_2 (값 30)에 비트 오염 발생 -> 255로 손상
    surgery_engine.inject_bit_rot(graph, "BYTE_2", 255)
    assert graph.nodes["BYTE_2"].payload["val"] == 255
    assert graph.nodes["BYTE_2"].payload.get("corrupted", False)

    # 인과적 자가 치유 실행 ("let it heal physically")
    healed_val = surgery_engine.self_heal_bit_rot(graph, "BYTE_2", invariant_parity_sum)

    assert healed_val == 30
    assert graph.nodes["BYTE_2"].payload["val"] == 30
    assert not graph.nodes["BYTE_2"].payload["corrupted"]
    assert graph.nodes["BYTE_2"].payload["healed"]
