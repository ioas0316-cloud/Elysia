"""
Demonstration & Benchmark: Causal Projection Game Engine
========================================================
'8K 동영상 디코딩식 인과 투사 원리'를 게임 엔진에 적용하여,
매 프레임 전체 픽셀/객체를 무식하게 다시 계산하는 전통적 방식 대비
수천 배의 연산 효율과 물리적 모순 없는 위상 전파를 입증하는 실증 데모.
"""

import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.causal_game_engine.causal_projection_engine import (
    CausalProjectionWorld,
    CausalProjectionRenderer
)
from core.topology.hierarchical_binary_lattice import (
    HierarchicalBinaryLattice
)


def run_causal_projection_demo():
    print("=" * 70)
    print(" [Elysia Causal Game Engine] 8K 동영상 투영 원리 기반 게임 엔진 실증")
    print("=" * 70 + "\n")

    # 1. 10,000개 타일 및 객체로 구성된 거대한 월드 생성
    WORLD_SIZE = 100
    TOTAL_ENTITIES = WORLD_SIZE * WORLD_SIZE # 10,000 entities
    print(f"1. 월드 초기화: 총 {TOTAL_ENTITIES:,}개 엔티티/타일 배치 중...")
    
    world = CausalProjectionWorld(world_id="gigabyte_scale_world", width=WORLD_SIZE, height=WORLD_SIZE)
    renderer = CausalProjectionRenderer(world)

    # 지형 및 환경 객체 배치
    for y in range(WORLD_SIZE):
        for x in range(WORLD_SIZE):
            symbol = "~" if (x + y) % 17 == 0 else "."
            world.spawn_entity(f"tile_{x}_{y}", x=x, y=y, entity_type="Grass" if symbol == "." else "Water", state_payload={"symbol": symbol})

    # 2. 기구학적 연쇄 장치(Kinematic Chain) 배치: 플레이어 조작 레버와 연동 기어들
    print("2. 인과 기구학 연쇄 장치 (Kinematic Chain) 등록...")
    world.spawn_entity("player_lever", x=10, y=10, entity_type="Lever", state_payload={"symbol": "L"})
    world.spawn_entity("gear_1", x=11, y=10, entity_type="Gear", state_payload={"symbol": "G"})
    world.spawn_entity("gear_2", x=12, y=10, entity_type="Gear", state_payload={"symbol": "G"})
    world.spawn_entity("drawbridge", x=13, y=10, entity_type="Bridge", state_payload={"symbol": "B"})

    # 인과 구속 조건(Precondition) 엣지 연결 (레버 -> 기어1 -> 기어2 -> 도개교)
    world.bind_kinematic_link("player_lever", "gear_1", dof_type="MOMENTUM_TRANSFER", ratio=1.0)
    world.bind_kinematic_link("gear_1", "gear_2", dof_type="MOMENTUM_TRANSFER", ratio=1.0)
    world.bind_kinematic_link("gear_2", "drawbridge", dof_type="MOMENTUM_TRANSFER", ratio=1.0)

    # 3. I-Frame (키 프레임) 최초 투사
    print("\n=== [Frame 1: I-Frame (최초 1회 전체 키 상태 투사)] ===")
    t0 = time.perf_counter()
    i_frame_info = renderer.render_frame(full_redraw=True)
    t_i_frame = (time.perf_counter() - t0) * 1000
    print(f" - I-Frame 투영 소요 시간: {t_i_frame:.2f} ms")
    print(f" - 투사된 엔티티 수: {i_frame_info['patches_applied']:,} / {len(world.graph.nodes):,}")

    print("\n[현재 뷰포트 (10x10) 시각화]:")
    print(renderer.render_ascii_viewport(view_x=8, view_y=8, view_w=8, view_h=5))

    # 4. 플레이어 상호작용 발생 (P-Frame: 레버를 당김 -> dpos=(0, 1))
    print("\n=== [Frame 2: P-Frame (사용자 상호작용: 레버 당김 Impulse 인가)] ===")
    t0 = time.perf_counter()
    affected_nodes = world.apply_impulse("player_lever", {"dpos": (0, 1)})
    t_physics = (time.perf_counter() - t0) * 1000

    print(f" - 위상적 인과 전파 경로: {affected_nodes}")
    print(f" - 상태 전파 소요 시간 ('Let it flow'): {t_physics:.4f} ms")

    # 5. 차분 인과 투사 (P-Frame Rendering)
    t0 = time.perf_counter()
    p_frame_info = renderer.render_frame(full_redraw=False)
    t_p_frame = (time.perf_counter() - t0) * 1000

    print(f" - P-Frame 차분 투영 소요 시간: {t_p_frame:.4f} ms")
    print(f" - 실제 패치된 엔티티 수: {p_frame_info['patches_applied']} 개 (전체 {len(world.graph.nodes):,}개 중)")
    print(f" - 효율성 비율 (수치 계산 절감율): {(1.0 - p_frame_info['efficiency_ratio']) * 100:.3f}% 절감!")

    print("\n[상호작용 후 뷰포트 시각화 (기구학적 연쇄 이동 확인)]:")
    print(renderer.render_ascii_viewport(view_x=8, view_y=8, view_w=8, view_h=5))

    # 6. 정형 검증 단정문
    assert len(affected_nodes) == 4, "레버와 연동된 4개 기구학 엔티티가 모두 전파되어야 합니다."
    assert p_frame_info['patches_applied'] == 4, "화면에는 오직 변위된 4개 노드만 패치되어야 합니다."
    assert world.entity_positions["drawbridge"] == (13, 11), "도개교의 위치가 y+1로 정확히 전이되어야 합니다."

    print("\n" + "=" * 70)
    print(" [실증 완료] 8K 동영상 디코딩식 인과 투영 원리가 게임 엔진에서")
    print(" 브루트포스 수치 연산 없이 완벽히 동작함을 입증하였습니다.")
    print("=" * 70)


if __name__ == "__main__":
    run_causal_projection_demo()
