import torch
import torch.nn as nn
import torch.nn.functional as F

from core.physics.continuous_thought_pipeline import (
    ContinuousThoughtPipeline,
    CombinedEnergyLoss,
    RotorSandwichFunctionNative
)


def test_rotor_sandwich_gradcheck():
    # 100% 미분 가능한 Clifford 샌드위치 연산의 Autograd GradCheck 검증
    v = torch.randn(2, 8, dtype=torch.float64, requires_grad=True)
    u = torch.randn(2, 8, dtype=torch.float64, requires_grad=True)
    w = torch.randn(2, 8, dtype=torch.float64, requires_grad=True)
    theta = torch.randn(2, 1, dtype=torch.float64, requires_grad=True)

    # Orthonormalize u and w to strictly satisfy geometry conditions for gradcheck stability
    with torch.no_grad():
        u.copy_(F.normalize(u, p=2, dim=-1))
        proj = (u * w).sum(dim=-1, keepdim=True) * u
        w.copy_(F.normalize(w - proj, p=2, dim=-1))

    res = torch.autograd.gradcheck(RotorSandwichFunctionNative.apply, (v, u, w, theta), eps=1e-6, atol=1e-4)
    assert res


def test_continuous_thought_pipeline_forward_and_backward():
    d_model = 128
    pipeline = ContinuousThoughtPipeline(d_model=d_model, n_steps=6)

    # Batch size 4, sequence length 32
    x_input = torch.rand(4, 32)
    x_target = torch.rand(4, 32)

    v_final, trajectory, theta_trajectory = pipeline(x_input)

    assert v_final.shape == (4, d_model)
    assert trajectory.shape == (4, 7, d_model)
    assert theta_trajectory.shape == (4, 6)

    # 초구면 등거리 보존 검증: 모든 trajectory 내의 벡터들의 Norm = 1.0 확인
    traj_norms = torch.norm(trajectory, p=2, dim=-1)
    assert torch.allclose(traj_norms, torch.ones_like(traj_norms), atol=1e-5)

    # CombinedEnergyLoss 포텐셜 에너지 융합 손실 함수 검증
    criterion = CombinedEnergyLoss(w_geodesic=1.0, w_resonance=0.5, w_smoothness=0.1)

    # Target을 생성하기 위한 No-Grad 프로젝션
    with torch.no_grad():
        v_target, _, _, _ = pipeline.encoder(x_target)

    loss, logs = criterion(
        v_final=v_final,
        v_target=v_target,
        trajectory=trajectory,
        theta_trajectory=theta_trajectory
    )

    assert loss.dim() == 0  # 스칼라 손실 확인
    assert "energy_total" in logs
    assert "e_loss_geodesic" in logs
    assert "reward_resonance" in logs
    assert "e_smoothness" in logs

    # 역전파 흐름성 테스트: Loss 로부터 모든 모델 가중치에 유효한 그라디언트가 완벽히 차오르는지 확인
    loss.backward()

    # Encoder의 1D 컨볼루션 가중치 그라디언트 존재 여부 검증
    for param in pipeline.parameters():
        if param.requires_grad:
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()
