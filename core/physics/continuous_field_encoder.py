import torch
import torch.nn as nn
import torch.nn.functional as F


class HarmonicFourierProjection(nn.Module):
    """
    이산 토큰 인덱스 대신 연속 입력 신호 s(t)를
    고차원 조화 푸리에 파동 스펙트럼 Φ(s)로 프로젝션하는 레이어.
    """
    def __init__(self, n_harmonics: int = 64, max_freq: float = 100.0):
        super().__init__()
        self.n_harmonics = n_harmonics
        # 기하급수적으로 증가하는 연속 주파수 기저 (학습 불필요한 기하학적 상수)
        frequencies = torch.exp(
            torch.linspace(0, torch.log(torch.tensor(max_freq)), n_harmonics)
        )
        self.register_buffer("frequencies", frequencies)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [Batch, Seq_Len] 또는 [Batch, Seq_Len, 1] 연속 값 (예: 0.0 ~ 1.0 범위의 float)
        Returns:
            [Batch, Seq_Len, 2 * n_harmonics] 연속 위상 파동
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # [B, L, 1]

        # [B, L, 1] * [1, 1, K] -> [B, L, K]
        angles = x * self.frequencies.view(1, 1, -1) * (2.0 * torch.pi)

        # [cos(ωs), sin(ωs)] 결합 -> [B, L, 2 * K]
        fourier_features = torch.cat([torch.cos(angles), torch.sin(angles)], dim=-1)
        return fourier_features


class ContinuousFieldEncoder(nn.Module):
    """
    연속 입력 파동을 초구면 S^{D-1} 위상 벡터 v0 및
    클리포드 회전 이중벡터 B0 = u0 ∧ w0, 회전각 θ0로 변환하는 연속장 인코더.
    """
    def __init__(
        self,
        d_model: int = 512,
        n_harmonics: int = 64,
        hidden_dim: int = 256,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.d_model = d_model
        self.eps = eps

        # 1. 연속 조화 푸리에 투영기
        self.fourier_proj = HarmonicFourierProjection(n_harmonics=n_harmonics)
        in_dim = 2 * n_harmonics

        # 2. 연속 상태 시간 적분 (Local Field Convolution)
        self.field_conv = nn.Sequential(
            nn.Conv1d(in_dim, hidden_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Conv1d(hidden_dim, d_model, kernel_size=5, padding=2),
            nn.GELU(),
        )

        # 3. 시간 적분 가중치 생성기 (Continuous Attention Pooling)
        self.temporal_pool = nn.Linear(d_model, 1)

        # 4. 초기 회전 평면 및 회전각 도출 프로젝터
        self.v_proj = nn.Linear(d_model, d_model)
        self.u_proj = nn.Linear(d_model, d_model)
        self.w_proj = nn.Linear(d_model, d_model)
        self.theta_proj = nn.Linear(d_model, 1)

    def _gram_schmidt_orthonormalize(
        self, u_raw: torch.Tensor, w_raw: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        u, w 벡터 집합에 대한 배치 단위 그람-슈미트 직교화:
        ⟨u, w⟩ = 0 및 ||u|| = ||w|| = 1 보장.
        """
        # u0 정규화
        u0 = F.normalize(u_raw, p=2, dim=-1, eps=self.eps)

        # w_raw에서 u0 방향 투영 성분 제거
        proj_u_w = (u0 * w_raw).sum(dim=-1, keepdim=True) * u0
        w_ortho = w_raw - proj_u_w

        # w0 정규화
        w0 = F.normalize(w_ortho, p=2, dim=-1, eps=self.eps)
        return u0, w0

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            x: [Batch, Seq_Len] - 연속적 입력 데이터 (0.0 ~ 1.0 정규화 신호)
        Returns:
            v0: [Batch, D] - 초구면 S^{D-1} 위 정규화된 초기 상태 벡터 (||v0|| = 1)
            u0: [Batch, D] - 초기 회전 평면 첫 번째 기저 벡터 (||u0|| = 1)
            w0: [Batch, D] - 초기 회전 평면 두 번째 기저 벡터 (||w0|| = 1, ⟨u0, w0⟩ = 0)
            theta0: [Batch, 1] - 초기 연속 회전 속도/회전각
        """
        # Step 1: 연속 조화 푸리에 투영 -> [B, L, 2*K]
        fourier_feats = self.fourier_proj(x)

        # Step 2: 연속 상태 공간 특징 추출 (Conv1D) -> [B, D, L]
        conv_input = fourier_feats.transpose(1, 2)
        field_feats = self.field_conv(conv_input).transpose(1, 2)  # [B, L, D]

        # Step 3: 연속 시간 필드 적분 (Temporal Field Pooling)
        weights = F.softmax(self.temporal_pool(field_feats), dim=1)  # [B, L, 1]
        x_integrated = (field_feats * weights).sum(dim=1)  # [B, D]

        # Step 4: 구면 위상 정규화 (Spherical Normalization -> S^{D-1})
        v_raw = self.v_proj(x_integrated)
        v0 = F.normalize(v_raw, p=2, dim=-1, eps=self.eps)

        # Step 5: 회전 이중벡터 B0 = u0 ∧ w0 추출 및 그람-슈미트 직교화
        u_raw = self.u_proj(v0)
        w_raw = self.w_proj(v0)
        u0, w0 = self._gram_schmidt_orthonormalize(u_raw, w_raw)

        # Step 6: 초기 회전각 θ0 (Softplus를 통한 양의 연속 회전 속도)
        theta0 = F.softplus(self.theta_proj(v0))

        return v0, u0, w0, theta0
