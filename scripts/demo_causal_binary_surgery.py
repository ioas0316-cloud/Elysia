"""
Demonstration: Binary Container Stripping & Causal World Surgery
================================================================
실제 구조화된 바이너리 파일을 해체하여 포맷 껍데기를 C(맥락)로 격리하고,
순수 불변 줄기(Stem)에 do-operator 외과적 수술 및 손상 비트 자가 치유를
수행한 후 100% 무손실 재합성을 증명하는 실증 스크립트.
"""

import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.topology.binary_container_stripper import BinaryContainerStripper
from core.topology.causal_world_surgery_engine import CausalWorldSurgeryEngine


def run_binary_surgery_demo():
    print("=" * 70)
    print(" [Elysia LOGOS] Binary Container Stripping & Causal Surgery Demo")
    print("=" * 70 + "\n")

    stripper = BinaryContainerStripper()
    surgery = CausalWorldSurgeryEngine()

    # 1. 원본 구조화된 바이너리 컨테이너 생성
    original_data = b"ELYSIAN_CAUSAL_INTELLIGENCE_INVARIANT_SEED_2026"
    raw_binary = stripper.create_sample_container(original_data)
    print(f"1. 원본 바이너리 컨테이너 생성 완료: 총 {len(raw_binary)} 바이트 (헤더 16B + 페이로드 {len(original_data)}B)")
    print(f" - Hex 미리보기: {raw_binary[:32].hex()} ...")

    # 2. 컨테이너 박리: 껍데기 메타데이터를 C로 격리하고 Stem 적출
    print("\n2. 컨테이너 껍데기 박리 (Container Stripping to C)...")
    t0 = time.perf_counter()
    graph = stripper.strip_and_parse(raw_binary)
    t_strip = (time.perf_counter() - t0) * 1000

    ctx = graph.context.environmental_constraints
    print(f" - 박리 소요 시간: {t_strip:.3f} ms")
    print(f" - 격리된 맥락(C): Magic={ctx['magic']}, Version={ctx['version']}, CRC={hex(ctx['stored_crc'])}")
    print(f" - 적출된 Stem 인과 노드: {len(graph.nodes)} 개, 엣지: {len(graph.edges)} 개")

    # 3. 100% 무손실 재합성 검증 (Roundtrip Test)
    print("\n3. 무손실 재합성(Resynthesis) 1:1 비트 일치 검증...")
    reconstructed = stripper.resynthesize(graph)
    assert reconstructed == raw_binary, "재합성된 바이너리가 원본과 1비트의 오차도 없이 동일해야 합니다."
    print(" - [검증 통과] 원본 바이너리 <== 100% Bit-Identical ==> 재합성 바이너리!")

    # 4. do-operator 인과 격자 수술 (Surgical Intervention)
    print("\n4. do-operator 인과 격자 수술 (Surgical Mutation on Stem)...")
    target_node = "BYTE_0"
    original_char = chr(graph.nodes[target_node].payload["val"])
    mutated_char = "X"
    print(f" - 수술 대상: {target_node} ('{original_char}' -> '{mutated_char}')")

    t0 = time.perf_counter()
    mutated_graph = surgery.apply_do_surgery(graph, target_node, ord(mutated_char))
    t_surgery = (time.perf_counter() - t0) * 1000

    mutated_binary = stripper.resynthesize(mutated_graph)
    print(f" - 수술 및 정합 재합성 소요 시간: {t_surgery:.3f} ms")
    print(f" - 수술 후 페이로드 헤드: {mutated_binary[16:26]}")
    assert mutated_binary[16:17] == b"X"

    # 5. 비트 오염(Bit-rot) 주입 및 인과적 자가 치유 (Self-Healing)
    print("\n5. 비트-롯(Bit-Rot) 훼손 주입 및 인과적 자가 치유 ('Let it heal physically')...")
    corrupt_target = "BYTE_10"
    original_val = graph.nodes[corrupt_target].payload["val"]
    invariant_parity_sum = sum(n.payload["val"] for n in graph.nodes.values())

    # 훼손 주입: 0xFF (255)로 강제 오염
    surgery.inject_bit_rot(graph, corrupt_target, 0xFF)
    print(f" - [오염 발생] {corrupt_target} 본래값: {original_val} ('{chr(original_val)}') -> 손상값: 255 (0xFF)")

    # 인과적 자가 치유 발동
    t0 = time.perf_counter()
    healed_val = surgery.self_heal_bit_rot(graph, corrupt_target, invariant_parity_sum)
    t_heal = (time.perf_counter() - t0) * 1000

    print(f" - [치유 완료] 보존 법칙 제약(C)에 의해 복원된 값: {healed_val} ('{chr(healed_val)}')")
    print(f" - 자가 치유 소요 시간: {t_heal:.4f} ms")
    assert healed_val == original_val, "인과적으로 복원된 값이 훼손 전 원본 값과 정확히 일치해야 합니다."

    # 6. 최종 재합성 검증
    healed_binary = stripper.resynthesize(graph)
    assert healed_binary == raw_binary, "치유 후 재합성된 바이너리가 초기 원본과 완벽히 일치해야 합니다."

    print("\n" + "=" * 70)
    print(" [LOGOS 실증 성공] 바이너리 컨테이너 박리, do-수술, 비트 자가 치유가")
    print(" 단 1비트의 손실이나 재인코딩 오차 없이 완벽히 수렴함을 증명하였습니다.")
    print("=" * 70)


if __name__ == "__main__":
    run_binary_surgery_demo()
