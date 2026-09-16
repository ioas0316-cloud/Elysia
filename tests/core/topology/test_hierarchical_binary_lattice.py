"""
Unit Tests: Hierarchical Binary Lattice & Causal Projection Engine
==================================================================
계층적 메모리/월드 격자 및 동영상 투영 기반 인과 렌더러 단위 테스트.
"""

import pytest
from core.topology.hierarchical_binary_lattice import (
    HierarchicalBinaryLattice,
    HierarchicalLevel
)
from modules.causal_game_engine.causal_projection_engine import (
    CausalProjectionWorld,
    CausalProjectionRenderer
)


def test_hierarchical_binary_lattice_skeleton():
    # 256KB 공간 (4KB 블록 x 16블록/청크 x 4청크)
    lattice = HierarchicalBinaryLattice(total_size_bytes=256 * 1024, block_size=4096, chunk_size=65536)

    assert len(lattice.compound_nodes) > 0
    # 청크 수 확인 (256KB / 64KB = 4청크)
    chunks = [n for n in lattice.compound_nodes.values() if n.level == HierarchicalLevel.CHUNK]
    assert len(chunks) == 4

    # 블록 수 확인 (256KB / 4KB = 64블록)
    blocks = [n for n in lattice.compound_nodes.values() if n.level == HierarchicalLevel.BLOCK]
    assert len(blocks) == 64


def test_perturb_offset_dirty_propagation():
    lattice = HierarchicalBinaryLattice(total_size_bytes=1024 * 1024, block_size=4096, chunk_size=65536)

    # 오프셋 5000 (Chunk 0, Block 1)에 자극 인가
    affected_path = lattice.perturb_offset(5000, new_value_payload=0xFF)

    assert "CHUNK_0" in affected_path
    assert "BLOCK_0_1" in affected_path
    assert len(lattice.dirty_nodes) == 2

    # Dirty 소진 확인
    deltas = lattice.consume_dirty_deltas()
    assert len(deltas) == 2
    assert len(lattice.dirty_nodes) == 0


def test_causal_projection_world_and_kinematic_flow():
    world = CausalProjectionWorld(width=50, height=50)

    # 1. 엔티티 배치: 레버(Lever)와 기어(Gear)
    lever = world.spawn_entity("lever_1", x=10, y=10, entity_type="Lever")
    gear = world.spawn_entity("gear_1", x=10, y=11, entity_type="Gear")

    # 2. 인과 기구학 연동 (레버 이동 시 기어 연동 이동)
    world.bind_kinematic_link(parent_id="lever_1", child_id="gear_1", dof_type="MOMENTUM_TRANSFER", ratio=1.0)

    # 3. 레버에 충격(Impulse) 인가 (x 방향 +2)
    affected = world.apply_impulse("lever_1", {"dpos": (2, 0)})

    # 레버뿐만 아니라 기구학적으로 결합된 기어까지 자동 전파 확인 ("Let it flow")
    assert "lever_1" in affected
    assert "gear_1" in affected
    assert world.entity_positions["lever_1"] == (12, 10)
    assert world.entity_positions["gear_1"] == (12, 11)


def test_differential_projection_efficiency():
    world = CausalProjectionWorld(width=100, height=100)
    renderer = CausalProjectionRenderer(world)

    # 1000개의 고정 엔티티 생성
    for i in range(1000):
        world.spawn_entity(f"entity_{i}", x=i % 100, y=i // 100, entity_type="Tree")

    # 1. I-Frame (키 프레임 전체 투사)
    frame_i = renderer.render_frame(full_redraw=True)
    assert frame_i["patches_applied"] == 1000
    assert frame_i["efficiency_ratio"] == 1.0

    # 2. 단 1개의 객체에만 플레이어 조작 발생
    world.apply_impulse("entity_42", {"dpos": (1, 1)})

    # 3. P-Frame (차분 투사: 8K 동영상 디코더 방식)
    frame_p = renderer.render_frame(full_redraw=False)
    # 1000개를 다시 계산하지 않고 오직 1개만 패치됨을 확인
    assert frame_p["patches_applied"] == 1
    assert frame_p["efficiency_ratio"] == 0.001
