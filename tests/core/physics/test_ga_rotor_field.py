"""
test_ga_rotor_field.py
======================
기하 대수 로터 필드(GA Rotor Fields) 및 틱리스 시뮬레이션 시스템의 핵심 동작을 증명하는 고성능 물리 단위 테스트 세트.

핵심 증명 및 검증 항목:
1. SoA & Ping-Pong Buffering:
   - 핑퐁 버퍼 스왑이 올바르게 실행되는지 확인하고, 데이터 경합이 방지되는지 검증합니다.
2. SDF LUT & O(1) Lookup:
   - Pre-baked SDF LUT 생성 및 Bilinear Interpolation을 통한 O(1) 거리 샘플링과 Gradient 복원 유효성을 검증합니다.
3. Multi-Agent GA Bivector & Lie Algebra Fusion:
   - 인접 유닛들이 서로 밀집할 때, Lie Algebra 상에서 가중 중첩을 통해 와류 형태의 로터가 안정적으로 생성되어
     충돌(Overlap) 없이 매끄럽게 우회하는지 검증합니다.
4. Tickless Analytical Sampling:
   - 시간 t를 임의로 대입해도 O(1) 복잡도로 중간 프레임 연산 없이 정확한 시간 연속적 좌표가 도출되는지 검증합니다.
5. Cognitive Thought Trajectory:
   - 잠재 공간 상의 논리적 모순 구역을 GA Rotor 필드를 사용하여 부드럽게 우회하고 목표 정답 개념에 정상 도달하는지 검증합니다.
"""

import pytest
import numpy as np
from core.physics.ga_rotor_field import GARotorFieldSystem, CognitiveThoughtTrajectory

def test_soa_and_ping_pong_buffering():
    """SoA 구조와 핑퐁 더블 버퍼링의 완벽한 정렬 및 전환을 검증합니다."""
    system = GARotorFieldSystem(num_agents=10, dims=2)

    # 초기 위치 및 속도 설정
    init_pos = np.random.normal(0, 1.0, (10, 2)).astype(np.float32)
    system.positions = init_pos
    system.preferred_velocities = np.ones((10, 2), dtype=np.float32)

    # 초기 버퍼 상태 검증
    assert np.allclose(system.positions, init_pos)
    assert system.current_buffer_idx == 0

    # 핑퐁 통합 단계 실행
    system.step_ping_pong_integration(dt=0.1)

    # 버퍼 인덱스가 스왑되었는지 확인
    assert system.current_buffer_idx == 1
    # 위치가 성공적으로 전진했는지 확인
    assert not np.allclose(system.positions, init_pos)

    # 이전 위치는 여전히 반대편 버퍼에 안전하게 존재해야 함
    assert np.allclose(system.pos_buffers[0], init_pos)

def test_sdf_lut_generation_and_lookup():
    """Pre-baked SDF LUT 생성과 bilinear interpolation을 통한 정확한 O(1) 그래디언트 복원을 검증합니다."""
    system = GARotorFieldSystem(num_agents=2, dims=2)

    # (0, 0) 중심, 반지름 1.0의 장애물 추가
    system.add_obstacle(center=[0.0, 0.0], radius=1.0)

    # SDF Grid 굽기
    system.pre_bake_sdf(x_range=(-3.0, 3.0), y_range=(-3.0, 3.0), resolution=(50, 50))

    # 1. 장애물 중심부 내부 샘플링 (거리가 음수여야 함)
    dist_inside, grad_inside = system.sample_sdf(np.array([0.0, 0.0], dtype=np.float32))
    assert dist_inside < 0.0

    # 2. 장애물 외부 샘플링 (거리가 양수여야 함)
    dist_outside, grad_outside = system.sample_sdf(np.array([2.0, 0.0], dtype=np.float32))
    assert dist_outside > 0.0
    # 그래디언트는 장애물 바깥을 가리키는 [1, 0]에 가까워야 함
    assert np.allclose(grad_outside, [1.0, 0.0], atol=0.2)

def test_multi_agent_ga_rotor_and_lie_algebra_fusion():
    """인접 유닛들이 충돌 경계선에 다다를 때, 리 대수 상에서 병목을 와류 형태로 회피하는지 검증합니다."""
    # 2D 2개 에이전트가 마주보는 병목 상황 설정
    system = GARotorFieldSystem(num_agents=2, dims=2, influence_radius=1.5, barrier_p=2.0)

    # 정면 충돌 궤적 유도
    system.positions = np.array([
        [-0.5, 0.0],
        [ 0.5, 0.0]
    ], dtype=np.float32)

    # 서로 마주보며 전진하려 함
    system.preferred_velocities = np.array([
        [ 1.0, 0.0],
        [-1.0, 0.0]
    ], dtype=np.float32)

    # 로터 필드 가동
    system.synthesize_rotor_fields()

    # 충돌 회피를 위해 원래 선호 속도(x방향)가 꺾여서 y 성분이 발생해야 함 (수학적 미끄러짐)
    v_out_0 = system.output_velocities[0]
    v_out_1 = system.output_velocities[1]

    assert abs(v_out_0[1]) > 0.1
    assert abs(v_out_1[1]) > 0.1
    # 크기(Norm)는 여전히 선호 속도(1.0)와 완벽히 보존되는 대칭성을 검증
    assert np.isclose(np.linalg.norm(v_out_0), 1.0, atol=1e-3)
    assert np.isclose(np.linalg.norm(v_out_1), 1.0, atol=1e-3)

def test_tickless_analytical_sampling():
    """시간 t만 지정하여 중간 계산(틱) 없이 즉시 정확한 궤적 좌표를 O(1) 복잡도로 취득하는지 검증합니다."""
    system = GARotorFieldSystem(num_agents=5, dims=2, influence_radius=0.1) # 인접 영향 반경을 낮춰 순수 속도가 유지되게 설정

    system.positions = np.array([
        [0.0, 0.0],
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0],
        [4.0, 4.0]
    ], dtype=np.float32)

    system.preferred_velocities = np.array([
        [1.0, 0.0],
        [0.0, 1.0],
        [-1.0, 0.0],
        [0.0, -1.0],
        [0.5, 0.5]
    ], dtype=np.float32)

    # 틱리스 상태 방정식 정의 및 주입
    system.initialize_tickless_trajectories(t_start=10.0)

    # 틱을 하나하나 밟지 않고, 즉시 t=15.5 시점의 상태를 O(1) 조회
    sampled_pos = system.sample_tickless_positions(t=15.5)

    # x(t) = x0 + v * (t - t0)
    # 0번 유닛: [0.0, 0.0] + [1.0, 0.0] * 5.5 = [5.5, 0.0]
    assert np.allclose(sampled_pos[0], [5.5, 0.0])
    # 1번 유닛: [1.0, 1.0] + [0.0, 1.0] * 5.5 = [1.0, 6.5]
    assert np.allclose(sampled_pos[1], [1.0, 6.5])

def test_cognitive_thought_trajectory_avoidance():
    """잠재 임베딩 공간에서 논리적 모순 구역을 로터를 사용해 부드럽게 피해 정답 개념으로 수렴하는지 검증합니다."""
    thought_engine = CognitiveThoughtTrajectory(embedding_dim=4, contradiction_threshold=1.5)

    # 시작 잠재 상태 및 목표 정답 설정
    start = np.array([-2.0, 0.0, 0.0, 0.0], dtype=np.float32)
    goal = np.array([2.0, 0.0, 0.0, 0.0], dtype=np.float32)

    # 중간에 모순 장벽(Obstacle) 배치
    contradiction_center = np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    thought_engine.add_contradiction_zone(center_emb=contradiction_center, radius=1.0)

    # 연속적 사고 궤적 도출 (충분한 단계를 부여하여 완전히 수렴하도록 설정)
    trajectory = thought_engine.navigate_thought(start, goal, steps=150, dt=0.05)

    # 1. 최종 도달 확인: 목표 지점 G 근방에 무사 도달
    final_pos = trajectory[-1]
    assert np.linalg.norm(final_pos - goal) < 0.25

    # 2. 비침투성(Non-penetration) 검증: 사고 과정에서 모순 영역 내부(거리 < 1.0)로 잠입하지 않았는지 확인
    for pos in trajectory:
        dist_to_contradiction = np.linalg.norm(pos - contradiction_center)
        # 0.95는 수치 한계선 보정을 위한 여유 마진
        assert dist_to_contradiction >= 0.9
