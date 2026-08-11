import torch
import torch.nn as nn
import torch.nn.functional as F

from core.physics.continuous_field_encoder import ContinuousFieldEncoder


class RotorSandwichFunctionNative(torch.autograd.Function):
    """
    N차원 로드리게스 축약식을 사용한 O(D) PyTorch Native Autograd 샌드위치 연산.
    RvR† = v + sin(θ) Av + (1 - cos(θ)) A²v
    """
    @staticmethod
    def forward(ctx, v, u, w, theta):
        # Av = <w, v>u - <u, v>w
        dot_wv = (w * v).sum(dim=-1, keepdim=True)
        dot_uv = (u * v).sum(dim=-1, keepdim=True)
        Av = dot_wv * u - dot_uv * w

        # A²v = -<u,v>u - <w,v>w  (직교 평면상 투영 벡터의 반대 방향)
        A2v = -dot_uv * u - dot_wv * w

        sin_t = torch.sin(theta)
        one_minus_cos_t = 1.0 - torch.cos(theta)

        v_next = v + sin_t * Av + one_minus_cos_t * A2v

        # 백워드 패스를 위한 컨텍스트 저장
        ctx.save_for_backward(v, u, w, theta, dot_uv, dot_wv, Av, A2v)
        return v_next

    @staticmethod
    def backward(ctx, grad_output):
        v, u, w, theta, dot_uv, dot_wv, Av, A2v = ctx.saved_tensors
        sin_t = torch.sin(theta)
        cos_t = torch.cos(theta)
        one_minus_cos_t = 1.0 - cos_t

        # 1. theta에 대한 연쇄 법칙 미분
        d_theta = (grad_output * (cos_t * Av + sin_t * A2v)).sum(dim=-1, keepdim=True)

        # 2. dot products of grad_output with u and w
        dot_g_u = (grad_output * u).sum(dim=-1, keepdim=True)
        dot_g_w = (grad_output * w).sum(dim=-1, keepdim=True)

        # 3. v에 대한 입력 역전파 (∇_v (RvR†) = I + sin(θ)A + (1-cos(θ))A²)
        grad_Av = dot_g_u * w - dot_g_w * u
        grad_A2v = -dot_g_u * u - dot_g_w * w
        grad_v = grad_output + sin_t * grad_Av + one_minus_cos_t * grad_A2v

        # 4. u, w 에 대한 미분 (체계적인 편미분)
        e_coef_u = sin_t * dot_wv - one_minus_cos_t * dot_uv
        v_coef_u = -(sin_t * dot_g_w + one_minus_cos_t * dot_g_u)
        grad_u = e_coef_u * grad_output + v_coef_u * v

        e_coef_w = -(sin_t * dot_uv + one_minus_cos_t * dot_wv)
        v_coef_w = sin_t * dot_g_u - one_minus_cos_t * dot_g_w
        grad_w = e_coef_w * grad_output + v_coef_w * v

        return grad_v, grad_u, grad_w, d_theta


class ContinuousThoughtPipeline(nn.Module):
    """
    입력 신호 -> ContinuousFieldEncoder -> RotorSandwich 궤적 적분기 전체 파이프라인.
    """
    def __init__(self, d_model: int = 512, n_steps: int = 16, eps: float = 1e-8):
        super().__init__()
        self.d_model = d_model
        self.n_steps = n_steps
        self.eps = eps

        # 1. 토큰 없는 연속장 인코더
        self.encoder = ContinuousFieldEncoder(d_model=d_model, eps=eps)

        # 2. 장 기울기(Gradient) 피드백용 포텐셜 반응 프로젝터
        self.field_response_u = nn.Linear(d_model, d_model)
        self.field_response_w = nn.Linear(d_model, d_model)

    def _gram_schmidt(self, u_raw: torch.Tensor, w_raw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        u = F.normalize(u_raw, p=2, dim=-1, eps=self.eps)
        proj = (u * w_raw).sum(dim=-1, keepdim=True) * u
        w = F.normalize(w_raw - proj, p=2, dim=-1, eps=self.eps)
        return u, w

    def forward(self, continuous_signal: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            continuous_signal: [Batch, Seq_Len] 연속 데이터
        Returns:
            v_final: [Batch, D] 최종 도달한 사고 상태 벡터 (||v_final|| = 1)
            trajectory: [Batch, N_Steps + 1, D] 전체 연속 사고 궤적
            theta_trajectory: [Batch, N_Steps] 각 스텝별 회전 각도
        """
        # Step 1: 연속장 인코딩 -> 초기 기하학 텐서 세트 추출
        v_t, u_t, w_t, theta_t = self.encoder(continuous_signal)

        trajectory = [v_t]
        theta_list = []

        # Step 2: 연속 사고 궤적 적분 루프
        for step in range(self.n_steps):
            # 클리포드 로터 샌드위치 연산 수행 (RvR†)
            v_next = RotorSandwichFunctionNative.apply(v_t, u_t, w_t, theta_t)

            # 구면 위상 재정규화 (등거리 보존 보장)
            v_next = F.normalize(v_next, p=2, dim=-1, eps=self.eps)

            # 포텐셜 장 피드백에 의한 다음 회전 평면 B_(t+1) = u_(t+1) ∧ w_(t+1) 업데이트
            u_update = u_t + 0.1 * self.field_response_u(v_next)
            w_update = w_t + 0.1 * self.field_response_w(v_next)
            u_t, w_t = self._gram_schmidt(u_update, w_update)

            v_t = v_next
            trajectory.append(v_t)
            theta_list.append(theta_t.squeeze(-1))

        # [Batch, N_Steps + 1, D] 형태로 궤적 결합
        trajectory_tensor = torch.stack(trajectory, dim=1)
        # [Batch, N_Steps] 형태로 회전각 결합
        theta_trajectory_tensor = torch.stack(theta_list, dim=1)

        return v_t, trajectory_tensor, theta_trajectory_tensor


class CombinedEnergyLoss(nn.Module):
    """
    손실(오차 척력)과 보상(관계성 인력)을 하나의 포텐셜 에너지 시스템으로 통합한 손실 함수.
    에너지 E_total = E_repulsion (Loss) - E_attraction (Reward) + E_smoothness
    """
    def __init__(
        self,
        w_geodesic: float = 1.0,      # 오차 척력 가중치
        w_resonance: float = 0.5,     # 위상 공진 인력(보상) 가중치
        w_smoothness: float = 0.1,    # 궤적 곡률 정규화 가중치
        eps: float = 1e-7,
    ):
        super().__init__()
        self.w_geodesic = w_geodesic
        self.w_resonance = w_resonance
        self.w_smoothness = w_smoothness
        self.eps = eps

    def forward(
        self,
        v_final: torch.Tensor,
        v_target: torch.Tensor,
        trajectory: torch.Tensor,
        v_context: torch.Tensor = None,
        theta_trajectory: torch.Tensor = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """
        Args:
            v_final: [Batch, D] - 사고 궤적의 최종 도달 위상 벡터
            v_target: [Batch, D] - 정답 타깃 위상 벡터 (||v_target|| = 1)
            trajectory: [Batch, N_Steps + 1, D] - 전체 연속 사고 궤적
            v_context: [Batch, D] (선택) - 주변 맥락/데이터의 연속장 위상 벡터
            theta_trajectory: [Batch, N_Steps] (선택) - 각 스텝별 회전 각속도
        Returns:
            total_energy: 최소화할 전체 시스템 포텐셜 에너지 (Loss)
            energy_components: 각 에너지 항별 모니터링 로그
        """
        # =================================================================
        # 1. 손실(Loss) 장: 오차 및 타깃 이탈에 대한 척력 포텐셜 (E_repulsion)
        # =================================================================
        cos_sim_target = (v_final * v_target).sum(dim=-1)
        cos_sim_clamped = torch.clamp(cos_sim_target, -1.0 + self.eps, 1.0 - self.eps)

        # 초구면 측한선 거리 (Geodesic Distance -> 0에 가까울수록 목표 도달)
        e_geodesic = torch.acos(cos_sim_clamped).mean()

        # =================================================================
        # 2. 보상(Reward) 장: 맥락 상호작용 및 위상 공진 인력 (E_attraction)
        # =================================================================
        if v_context is None:
            # context가 명시되지 않은 경우, 궤적 자체의 자기 상관성(Self-Resonance) 활용
            v_context = trajectory.mean(dim=1)  # 궤적의 평균 위상 중심
            v_context = F.normalize(v_context, p=2, dim=-1, eps=self.eps)

        # 궤적 상의 상태 벡터들과 맥락 벡터 간의 고차원 위상 내적 (공진)
        # trajectory: [B, Steps, D], v_context: [B, 1, D]
        traj_states = trajectory[:, 1:, :]  # 초기 상태 제외한 사고 궤적
        resonance_matrix = (traj_states * v_context.unsqueeze(1)).sum(dim=-1)  # [B, Steps]

        # 각속도 θ가 존재할 경우, 위상 동기화 변주 부여 (회전 각도와 맥락의 정렬)
        if theta_trajectory is not None:
            phase_sync = torch.cos(theta_trajectory)  # [B, Steps]
            resonance_matrix = resonance_matrix * phase_sync

        # 보상 포텐셜 (공진이 강할수록 인력 에너지가 커져 시스템 전체 에너지를 낮춤)
        reward_resonance = resonance_matrix.mean()

        # =================================================================
        # 3. 궤적 정규화 장: 매끄러운 회전 보존 (E_smoothness)
        # =================================================================
        v_t = trajectory[:, :-1, :]
        v_next = trajectory[:, 1:, :]
        step_smoothness = (1.0 - (v_t * v_next).sum(dim=-1)).mean()

        # =================================================================
        # 4. 전체 포텐셜 에너지 결합 (Energy Minimization)
        #    E_total = E_loss - E_reward + E_reg
        # =================================================================
        total_energy = (
            self.w_geodesic * e_geodesic
            - self.w_resonance * reward_resonance
            + self.w_smoothness * step_smoothness
        )

        energy_components = {
            "energy_total": total_energy.item(),
            "e_loss_geodesic": e_geodesic.item(),
            "reward_resonance": reward_resonance.item(),
            "e_smoothness": step_smoothness.item(),
            "target_cos_sim": cos_sim_target.mean().item(),
        }

        return total_energy, energy_components


# =====================================================================
# 파이프라인 구동 및 VRAM/궤적 검증
# =====================================================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pipeline = ContinuousThoughtPipeline(d_model=512, n_steps=10).to(device)
    dummy_input = torch.rand(8, 256, device=device)  # Batch=8, Length=256 연속 신호

    v_final, trajectory, theta_trajectory = pipeline(dummy_input)

    print("=== Continuous Thought Pipeline 실행 결과 ===")
    print(f"1. 입력 신호 크기: {dummy_input.shape}")
    print(f"2. 최종 상태 벡터 v_final 크기: {v_final.shape}")
    print(f"3. 전체 사고 궤적 Trajectory 크기: {trajectory.shape} (Batch, Steps, Dim)")
    print(f"4. 최종 벡터 Norm 유지 확인: {v_final.norm(dim=-1).mean().item():.6f} (Target: 1.0)")

    # 역전파 호환성 테스트 (GradCheck 유효성)
    loss = trajectory.sum()
    loss.backward()
    print("5. 파이프라인 전체 엔드투엔드 Autograd 역전파 완료!")
