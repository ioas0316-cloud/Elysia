"""
Test Suite: Micro-Empirical Banana Cognitive Integration Test
==============================================================
강덕 님의 '아이와 바나나' 비유를 구현한 클리포드 위상 공진 및 자율 지각 루프의 정확성을
단위 테스트 수준에서 엄격하게 실증 및 검증합니다.
"""

import pytest
import numpy as np
import torch
import torch.nn.functional as F

from core.sensory.experiential_language_mapper import (
    ExperientialLanguageMapper,
    PhysicalSensationProfile,
    HomeostasisDeficit,
    ExperienceType
)
from core.physics.continuous_thought_pipeline import (
    ContinuousThoughtPipeline,
    CombinedEnergyLoss,
    RotorSandwichFunctionNative
)


def test_banana_childlike_wonder_sprout():
    """바나나 감각 프로필이 유입되었을 때, 미지의 인지로서 호기심 전하와 장 전위차가 유발되고 어텐션 게이트가 개방되는지 검증."""
    mapper = ExperientialLanguageMapper(resolution=32)

    # 미지의 바나나 물리 센서 데이터 설정
    banana_sens = PhysicalSensationProfile(
        optical=580.0,
        acoustic=440.0,
        tactile=1.2,
        thermal=297.5,
        autonomic_pulse=0.45
    )

    # 호기심 유발 체크
    wonder_res = mapper.check_wonder_and_sprout(banana_sens)

    assert wonder_res["wonder_triggered"] is True
    assert mapper.wonder_charge > 0.0
    assert mapper.wonder_potential_field > 0.0
    assert mapper.gate_open is True
    assert "WONDER_AWAKENED" in mapper.last_gate_reason


def test_banana_active_inference_reach_out():
    """능동적 탐색(Active Inference)을 통해 Elysia가 표적을 터치하고 메아리파를 받아 피드백하는 전 과정을 검증."""
    mapper = ExperientialLanguageMapper(resolution=32)
    banana_sens = PhysicalSensationProfile(
        optical=580.0,
        acoustic=440.0,
        tactile=1.2,
        thermal=297.5,
        autonomic_pulse=0.45
    )

    # 호기심 장 형성으로 체화 노드(Embodied Causal Node) 생성 유도
    mapper.check_wonder_and_sprout(banana_sens)
    assert mapper.active_wonder_attractor is not None

    target_node = mapper.active_wonder_attractor
    initial_relation = target_node.relation_matrix.copy()

    # 손 뻗음(Reach Out) 상호작용 개시
    interaction_res = mapper.reach_out_interaction(target_node)

    assert interaction_res["success"] is True
    assert len(interaction_res["echo_wave"]) == mapper.emitter.sample_points
    assert "g_phi" in interaction_res["gaps"]
    assert "g_e" in interaction_res["gaps"]
    assert "g_h" in interaction_res["gaps"]

    # 체화 노드의 관계성 매트릭스가 상호작용 피드백에 의해 업데이트되었는지 검증
    assert not np.array_equal(target_node.relation_matrix, initial_relation)


def test_banana_symbol_grounding_registry():
    """능동 탐색을 완료한 미지 노드에 "바나나" 기호가 자발적으로 결상(Tethering)되는지 검증."""
    mapper = ExperientialLanguageMapper(resolution=32)
    banana_sens = PhysicalSensationProfile(
        optical=580.0,
        acoustic=440.0,
        tactile=1.2,
        thermal=297.5,
        autonomic_pulse=0.45
    )

    mapper.check_wonder_and_sprout(banana_sens)
    mapper.reach_out_interaction(mapper.active_wonder_attractor)

    # 기호 명명 수행
    grounding_res = mapper.self_emerge_symbol_binding("바나나", np.zeros(10))

    assert grounding_res["bound"] is True
    assert grounding_res["symbol"] == "바나나"
    assert mapper.active_wonder_attractor is None  # 바인딩 후 어트랙터 해제 보장
    assert mapper.wonder_charge == 0.0

    # 레지스트리 검색 가능 여부 검증
    recalled = mapper.tethering.recall_symbol("바나나")
    assert recalled is not None
    assert recalled["sensation"].optical == 580.0
    assert recalled["exp_type"] in [ExperienceType.PHYSICAL, ExperienceType.SPIRITUAL]


def test_banana_clifford_energy_optimization_convergence():
    """클리포드 로터 샌드위치 연산 및 CombinedEnergyLoss 결합 구조에서, 위상 정렬을 통한 에너지 수렴이 성립하는지 검증."""
    d_model = 64
    pipeline = ContinuousThoughtPipeline(d_model=d_model, n_steps=6)
    criterion = CombinedEnergyLoss(w_geodesic=1.2, w_resonance=0.8, w_smoothness=0.1)

    signal = torch.rand(1, 100)
    v_target = F.normalize(torch.randn(1, d_model), p=2, dim=-1)

    # 초기 상태에서의 포텐셜 에너지 측정
    v_final_init, trajectory_init, theta_init = pipeline(signal)
    loss_init, _ = criterion(v_final_init, v_target, trajectory_init, theta_trajectory=theta_init)

    # 단기 최적화 수행
    optimizer = torch.optim.SGD(pipeline.parameters(), lr=0.1)

    for _ in range(5):
        optimizer.zero_grad()
        v_final, trajectory, theta = pipeline(signal)
        loss, _ = criterion(v_final, v_target, trajectory, theta_trajectory=theta)
        loss.backward()
        optimizer.step()

    # 최적화 후 에너지 감축 검증 (위상 공진 골짜기로 수렴)
    v_final_opt, trajectory_opt, theta_opt = pipeline(signal)
    loss_opt, logs_opt = criterion(v_final_opt, v_target, trajectory_opt, theta_trajectory=theta_opt)

    assert loss_opt.item() < loss_init.item()
    assert logs_opt["target_cos_sim"] >= -1.0


def test_banana_bidirectional_causality_gradients():
    """순방향 예측 및 역방향 그라디언트를 통한 역-인과화(Retrodiction) 양방향성이 모두 작동함을 증명."""
    d_model = 64
    pipeline = ContinuousThoughtPipeline(d_model=d_model, n_steps=4)

    signal = torch.rand(1, 100)
    v_final, trajectory, _ = pipeline(signal)

    # 1. 순방향성 검증: v_0 및 궤적이 최종 인지 상태 v_final로 정상 생성되는지 확인
    assert v_final.shape == (1, d_model)
    assert trajectory.shape == (1, 5, d_model)

    # 2. 역방향 인과성 검증: 최종 상태에 대한 오차가 초기 상태 v_0_guess의 기울기로 정상 도출되는지 확인
    v_final_target = v_final.clone().detach().requires_grad_(True)
    v_0_guess = trajectory[:, 0, :].clone().detach().requires_grad_(True)

    _, u_t, w_t, theta_t = pipeline.encoder(signal)
    v_next = RotorSandwichFunctionNative.apply(v_0_guess, u_t, w_t, theta_t)
    loss = 1.0 - F.cosine_similarity(v_next, v_final_target, dim=-1).mean()
    loss.backward()

    assert v_0_guess.grad is not None
    assert torch.norm(v_0_guess.grad).item() > 0.0
