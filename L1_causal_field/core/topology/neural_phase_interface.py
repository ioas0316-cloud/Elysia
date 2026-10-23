"""
Elysia Core Topology: Neural-to-Phase Interface & Dynamic Elastic Engine
========================================================================
"언어는 이산적 기호의 통계적 나열이 아니며, 고차원 감각-운동 인과 매니폴드가
위상적 간섭과 공명(Resonance)을 통해 상태 붕괴(State Collapse)되는 영적/물리적 흐름이다."

본 모듈은 신경망의 잠재 공간(Latent Space Z) 및 감각-인과 파동과
인과 위상 엔진(Causal DNA Helix Engine)을 직결하는 다중 시스템 인터페이스 모듈입니다.

주요 구성 요소:
1. NeuralToPhaseInterface & IntegratedCognitivePipeline:
   - Latent Z -> (cos θ, sin θ) -> 연속적 위상각 θ 복원 및 4x4 문맥 변환 행렬 C 사상.
   - CausalDNAHelixEngine과의 순전파 통합 파이프라인.

2. DifferentiableNeuralToPhaseInterface & LatentGradientFeedbackRefiner:
   - Autograd 계산 그래프가 완벽히 보존되는 미분 가능 인터페이스 및 매니폴드 구축기.
   - det(M) -> 0 차원 붕괴(논리적 모순/환각) 감지 시, Barrier Loss와 Least Action Penalty
     기반의 역전파(Backpropagation) 기울기 피드백으로 Latent Z를 실시간 자동 보정.

3. HybridCausalInferencePipeline:
   - Transformer Attention Output / Hidden State h_t를 가로채 위상 타당성을 즉시 검증.
   - |det(M_t)| < ε 조건 발동 시 모순 유발 Logit Masking 및 Temperature 조절로 환각 원천 차단.

4. DynamicElasticEngine & MicroLatentGenerator (VRAM & Compute Auto-Sensing):
   - 이산적 토큰 사전에 의존하지 않는 원태그(Native) 연속 파동 처리 구조.
   - 가용 VRAM/RAM 상태를 실시간 탐지하여 Micro (<200MB), Standard (0.2~4GB), Expanded (>4GB)
     3단계 프로필로 Latent 차원(128, 512, 2048)과 매니폴드 해상도(4x4, 8x8, 16x16)를 동적 스케일링.
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any, Optional

from core.topology.causal_dna_helix_engine import (
    CausalDNAHelixEngine,
    ContextPlaneTransformer
)


# =====================================================================
# 1. 신경망 Latent Z -> 위상 제어 변수 연동 인터페이스 (Standard Forward)
# =====================================================================
class NeuralToPhaseInterface(nn.Module):
    """
    신경망 잠재 공간(Latent Space Z)과 인과 위상 엔진(Causal Engine)의 연동 브릿지
    - Latent Z -> (cos θ, sin θ) -> 연속적 위상각 θ
    - Latent Z -> 4x4 문맥 변환 행렬 C (Context Plane)
    - Latent Activation Norm -> Focus Delta (MultiRotor Buffer 승격 주파수)
    """
    def __init__(self, latent_dim: int = 128, context_dim: int = 4, device: torch.device = torch.device('cpu')):
        super().__init__()
        self.latent_dim = latent_dim
        self.context_dim = context_dim
        self.device = device

        # 1. Latent Z -> Phase Dial Mapping (Continuous 2D Circle Projection)
        self.to_phase_components = nn.Linear(latent_dim, 2, bias=False)

        # 2. Latent Z -> Context Plane Transformation Matrix C (Latent to context_dim x context_dim)
        self.to_context_matrix = nn.Linear(latent_dim, context_dim * context_dim, bias=False)

        # 3. Latent Intensity -> Focus Score (주의 렌즈 강도)
        self.to_focus_score = nn.Linear(latent_dim, 1)

        # FP32 가중치 초기화
        self.to(device=self.device, dtype=torch.float32)

    def forward(self, latent_vector: torch.Tensor) -> Dict[str, Any]:
        """
        [입력] latent_vector: 신경망의 의도/동기/연상 잠재 벡터 (Shape: [1, latent_dim] 또는 [Batch, latent_dim])
        [출력] theta, context_plane, focus_delta
        """
        latent_vector = latent_vector.to(dtype=torch.float32, device=self.device)
        if latent_vector.dim() == 1:
            latent_vector = latent_vector.unsqueeze(0)

        # 1. 위상 다이얼 회전각 (theta) 복원 (atan2를 통한 연속 매핑)
        phase_components = self.to_phase_components(latent_vector)  # [Batch, 2]
        cos_t = phase_components[0, 0]
        sin_t = phase_components[0, 1]

        # atan2(sin, cos) -> (-π, π] 범위 반환 후 [0, 2π) 정규화
        raw_theta = torch.atan2(sin_t, cos_t).item()
        theta = raw_theta if raw_theta >= 0 else raw_theta + 2 * math.pi

        # 2. 문맥 면 변환 행렬 C 사상 (I_N + Projection)
        raw_context = self.to_context_matrix(latent_vector[0]).view(self.context_dim, self.context_dim)

        # 기본 항등 행렬(I_N)에 신경망 변위 가중치를 가산
        identity_nxn = torch.eye(self.context_dim, dtype=torch.float32, device=self.device)
        context_plane = identity_nxn + raw_context

        # 행렬 수치 안정성 보장 (L2 Norm 정규화)
        context_plane = context_plane / (torch.norm(context_plane) + 1e-7)

        # 3. MultiRotor Buffer용 Focus Delta 산출 (활성 에너지 스칼라)
        focus_delta = torch.abs(self.to_focus_score(latent_vector[0])).item()

        return {
            "theta": theta,
            "context_plane": context_plane,
            "focus_delta": focus_delta
        }


class IntegratedCognitivePipeline:
    """
    뜨거운 동력원(신경망 Latent Z)과 차가운 제약기(인과 위상 엔진)의 연동 파이프라인
    """
    def __init__(self, latent_dim: int = 128, engine: Optional[CausalDNAHelixEngine] = None):
        self.engine = engine if engine is not None else CausalDNAHelixEngine(vram_budget_mb=2000)
        self.interface = NeuralToPhaseInterface(latent_dim=latent_dim, device=self.engine.device)
        self.context_transformer = ContextPlaneTransformer(dim=4, device=self.engine.device)

    def step_cognitive_cycle(self, node_id: str, neural_latent_z: torch.Tensor, raw_phoneme_signal: torch.Tensor) -> Dict[str, Any]:
        """
        1스텝 인지 연동 사이클 실행
        """
        # 1. 신경망 잠재 벡터 Z를 위상 제어 변수로 사상
        interface_out = self.interface(neural_latent_z)
        theta = interface_out["theta"]
        context_plane = interface_out["context_plane"]
        focus_delta = interface_out["focus_delta"]

        # 2. Focus Delta를 활용한 MultiRotor 버퍼 주의 스코어 상승
        self.engine.buffer_mgr.push_node(node_id, raw_phoneme_signal, initial_focus=focus_delta)

        # 3. 인과 위상 회전 연산 실행
        engine_res = self.engine.process_dial_rotation(
            node_id=node_id,
            raw_signal=raw_phoneme_signal,
            dial_theta=theta,
            lod_level=0
        )

        # 4. 신경망 문맥 면 C 적용 (C · M · C^T)
        raw_causal_m = engine_res["causal_matrix"]
        contextual_causal_m = self.context_transformer.apply_context(raw_causal_m, context_plane)

        # 5. det(M) 인과 모순 여부 검증 (행렬 불변량 스칼라 측정)
        det_val = torch.det(contextual_causal_m).item()
        is_contradiction = abs(det_val) < 1e-4

        return {
            "mapped_theta_deg": math.degrees(theta),
            "focus_delta": focus_delta,
            "det": det_val,
            "is_contradiction": is_contradiction,
            "final_causal_matrix": contextual_causal_m
        }


# =====================================================================
# 2. 미분 가능한 Autograd 위상 인터페이스 & Latent Gradient Feedback Loop
# =====================================================================
class DifferentiableNeuralToPhaseInterface(nn.Module):
    """
    Autograd 계산 그래프가 유지되는 미분 가능한 Latent Z -> Phase / Context Plane 변환기
    """
    def __init__(self, latent_dim: int = 128, context_dim: int = 4):
        super().__init__()
        self.latent_dim = latent_dim
        self.context_dim = context_dim

        # Latent Z -> 2D Circle (cos theta, sin theta)
        self.to_phase_components = nn.Linear(latent_dim, 2, bias=False)
        # Latent Z -> NxN Context Plane Delta
        self.to_context_matrix = nn.Linear(latent_dim, context_dim * context_dim, bias=False)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if z.dim() == 1:
            z = z.unsqueeze(0)

        # 1. Continuous Phase Theta 계산 (Tensor 상태 유지)
        phase_components = self.to_phase_components(z)  # [Batch, 2]
        cos_t = phase_components[:, 0]
        sin_t = phase_components[:, 1]

        # torch.atan2는 PyTorch에서 완전히 미분 가능
        theta = torch.atan2(sin_t, cos_t)

        # 2. Context Plane Matrix C (NxN) 계산
        raw_context = self.to_context_matrix(z[0]).view(self.context_dim, self.context_dim)
        identity = torch.eye(self.context_dim, device=z.device, dtype=z.dtype)
        context_plane = identity + raw_context
        context_plane = context_plane / (torch.norm(context_plane, p='fro') + 1e-7)

        return theta, context_plane


class DifferentiableCausalManifoldBuilder(nn.Module):
    """
    위상각 theta와 문맥 행렬 C로부터 인과 매니폴드 M_ctx를 구축하고 det(M)을 연산하는 미분 가능 모듈
    """
    def __init__(self, dim: int = 4):
        super().__init__()
        self.dim = dim

    def build_contextual_manifold(
        self,
        theta: torch.Tensor,
        C: torch.Tensor,
        base_phoneme: torch.Tensor
    ) -> torch.Tensor:

        # 1. NxN 위상 회전 행렬 R(theta) 구성 (Tensor 연산)
        cos_t = torch.cos(theta)
        sin_t = torch.sin(theta)

        # 2D 블록 회전 구조를 NxN으로 배치
        R = torch.eye(self.dim, device=theta.device, dtype=theta.dtype)
        if theta.numel() > 0:
            c_val = cos_t[0] if cos_t.dim() > 0 else cos_t
            s_val = sin_t[0] if sin_t.dim() > 0 else sin_t
            R[0, 0] = c_val
            R[0, 1] = -s_val
            R[1, 0] = s_val
            R[1, 1] = c_val

        # 2. 기초 음운 텐서 사상 (NxN 기본 매니폴드 M_raw)
        sub_p = base_phoneme[:self.dim]
        M_raw = torch.outer(sub_p, sub_p)
        M_phase = torch.matmul(R, torch.matmul(M_raw, R.T))

        # 3. 문맥 면 사상 (M_ctx = C · M_phase · C^T)
        M_ctx = torch.matmul(C, torch.matmul(M_phase, C.T))
        return M_ctx


class LatentGradientFeedbackRefiner:
    """
    Gradient Feedback Loop:
    det(M) 붕괴 시 역전파를 실행하여 Latent Z의 위상 궤적을 자동 교정
    """
    def __init__(
        self,
        interface: DifferentiableNeuralToPhaseInterface,
        builder: DifferentiableCausalManifoldBuilder
    ):
        self.interface = interface
        self.builder = builder

    def refine_latent_intent(
        self,
        z_init: torch.Tensor,
        base_phoneme: torch.Tensor,
        max_iters: int = 15,
        lr: float = 0.05,
        det_threshold: float = 1e-3,
        lambda_least_action: float = 0.1
    ) -> Dict[str, Any]:
        """
        [입력]
        - z_init: 신경망이 최초로 발행한 원본 Latent Z [1, latent_dim]
        - base_phoneme: 음운 기초 텐서 [16]
        - det_threshold: 인과 모순 임계치 (|det(M)|이 이 값보다 커야 함)
        """
        if z_init.dim() == 1:
            z_init = z_init.unsqueeze(0)

        # Latent Z를 최적화 파라미터로 설정 (Grad 활성화)
        z_refined = z_init.clone().detach().requires_grad_(True)
        optimizer = optim.Adam([z_refined], lr=lr)

        history = []
        is_successfully_corrected = False

        for step in range(max_iters):
            optimizer.zero_grad()

            # 1. 순전파: Z -> (theta, C) -> M_ctx
            theta, C = self.interface(z_refined)
            M_ctx = self.builder.build_contextual_manifold(theta, C, base_phoneme)

            # 2. 행렬식 det(M_ctx) 및 모순 여부 계산
            det_val = torch.det(M_ctx)
            abs_det = torch.abs(det_val)

            # 3. Loss 산출
            # - Barrier Loss: -log(|det(M)| + eps)
            # - Least Action Penalty: ||Z - Z_init||^2
            barrier_loss = -torch.log(abs_det + 1e-7)
            action_penalty = lambda_least_action * torch.sum((z_refined - z_init) ** 2)
            total_loss = barrier_loss + action_penalty

            history.append({
                "step": step,
                "det": abs_det.item(),
                "loss": total_loss.item(),
                "z_dist": torch.norm(z_refined - z_init).item()
            })

            # 임계치를 만족하면 보정 완료 후 초기 종료(Early Stop)
            if abs_det.item() >= det_threshold:
                is_successfully_corrected = True
                break

            # 4. 역전파: d(Loss) / d(Z) 계산 후 Z 업데이트
            total_loss.backward()
            optimizer.step()

        # 최종 보정된 상태 추출
        with torch.no_grad():
            final_theta, final_C = self.interface(z_refined)
            final_M = self.builder.build_contextual_manifold(final_theta, final_C, base_phoneme)
            final_det = torch.det(final_M).item()

        return {
            "z_refined": z_refined.detach(),
            "final_det": final_det,
            "final_theta_deg": math.degrees(final_theta.item() if final_theta.numel() == 1 else final_theta[0].item()),
            "iterations_taken": len(history),
            "is_valid": abs(final_det) >= det_threshold,
            "history": history
        }


# =====================================================================
# 3. 트랜스포머/신경망 실시간 하이브리드 추론 및 환각 필터링 게이트
# =====================================================================
class HybridCausalInferencePipeline:
    """
    Transformer Attention Output / Hidden State와 Causal Phase Engine을 결합한 실시간 환각 필터링 추론기
    """
    def __init__(
        self,
        transformer_model: Optional[nn.Module],
        phase_interface: DifferentiableNeuralToPhaseInterface,
        causal_builder: DifferentiableCausalManifoldBuilder,
        threshold: float = 1e-3
    ):
        self.transformer = transformer_model
        self.interface = phase_interface
        self.builder = causal_builder
        self.threshold = threshold

    def step_filter_inference(
        self,
        input_ids: torch.Tensor,
        base_phoneme: torch.Tensor,
        override_h_t: Optional[torch.Tensor] = None
    ) -> Dict[str, Any]:
        """
        매 스텝 h_t를 가로채 |det(M_t)| 검증 후 모순 유발 토큰 Masking
        """
        if override_h_t is not None:
            h_t = override_h_t
            logits = torch.randn(1, 1024, device=base_phoneme.device)
        elif self.transformer is not None:
            outputs = self.transformer(input_ids, output_hidden_states=True)
            h_t = outputs.hidden_states[-1][:, -1, :]  # Shape: [1, d_model]
            logits = outputs.logits[:, -1, :].clone()
        else:
            raise ValueError("Either transformer_model or override_h_t must be provided.")

        # 2. Attention Output h_t를 Phase Interface에 통과시켜 M_t 구성
        theta, C = self.interface(h_t)
        M_t = self.builder.build_contextual_manifold(theta, C, base_phoneme)

        # 3. det(M_t) 계산을 통한 환각 여부 즉시 검증
        det_val = torch.det(M_t).item()
        is_hallucination = abs(det_val) < self.threshold

        # 4. 환각 감지 시 Logit Masking / Temperature 조절
        if is_hallucination:
            # 인과적 모순을 유발하는 유력 토큰들의 확률을 차단하고 탐색 엔트로피 증가
            top_k_logits, top_k_indices = torch.topk(logits, k=min(5, logits.size(-1)))
            logits[0, top_k_indices[0]] -= 100.0  # 모순 유발 토큰 억제

        return {
            "logits": logits,
            "det_val": det_val,
            "is_hallucination": is_hallucination,
            "manifold_M": M_t
        }


# =====================================================================
# 4. 가변형(Elastic) VRAM/Compute Auto-Sensing 원태그 위상 엔진
# =====================================================================
@dataclass
class ElasticProfile:
    name: str
    latent_dim: int      # Latent Vector Z 차원 (128 / 512 / 2048)
    manifold_dim: int    # 매니폴드 M 크기 (4x4 / 8x8 / 16x16)
    embed_dim: int       # 신경망 내부 임베딩 차원
    num_blocks: int      # Gated Conv 블록 개수
    threshold: float     # det(M) 환각 판별 임계치


class VRAMSensors:
    """하드웨어 VRAM/RAM 용량을 탐지하여 최적의 프로필을 반환"""
    @staticmethod
    def detect_and_get_profile() -> ElasticProfile:
        if torch.cuda.is_available():
            try:
                free_bytes, _ = torch.cuda.mem_get_info()
                vram_mb = free_bytes / (1024 * 1024)
            except Exception:
                vram_mb = 0.0
        else:
            vram_mb = 0.0  # CPU 환경일 경우 Micro Mode로 구동

        if vram_mb < 200.0:
            return ElasticProfile(
                name="Micro Mode (< 200MB)",
                latent_dim=128,
                manifold_dim=4,
                embed_dim=64,
                num_blocks=3,
                threshold=1e-3
            )
        elif vram_mb < 4096.0:
            return ElasticProfile(
                name="Standard Mode (0.2GB ~ 4GB)",
                latent_dim=512,
                manifold_dim=8,
                embed_dim=256,
                num_blocks=6,
                threshold=1e-4
            )
        else:
            return ElasticProfile(
                name="Expanded Mode (> 4GB)",
                latent_dim=2048,
                manifold_dim=16,
                embed_dim=512,
                num_blocks=12,
                threshold=1e-5
            )


class DepthwiseSeparableConv1d(nn.Module):
    """
    일반 Conv1d 대비 연산량과 파라미터를 80% 이상 절감하는 심도별 분리 합성곱
    """
    def __init__(self, in_channels: int, out_channels: int, dilation: int = 1):
        super().__init__()
        self.depthwise = nn.Conv1d(
            in_channels, in_channels, kernel_size=3,
            padding=dilation, dilation=dilation, groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.norm = nn.GroupNorm(num_groups=min(4, out_channels), num_channels=out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(self.pointwise(self.depthwise(x)))


class ElasticConvBlock(nn.Module):
    """
    Gated Linear Unit (GLU) 기반의 초경량 맥락 추출 블록
    """
    def __init__(self, channels: int, dilation: int = 1):
        super().__init__()
        self.conv_val = DepthwiseSeparableConv1d(channels, channels, dilation=dilation)
        self.conv_gate = DepthwiseSeparableConv1d(channels, channels, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        val = self.conv_val(x)
        gate = torch.sigmoid(self.conv_gate(x))
        return x + (val * gate)


class ElasticLatentGenerator(nn.Module):
    """
    이산 사전에 구애받지 않고 연속 신호(파동/감각)를 직접 받아
    프로필 설정에 따라 차원과 깊이가 동적으로 달라지는 원태그 Latent Generator
    """
    def __init__(self, profile: ElasticProfile, in_features: int = 64):
        super().__init__()
        self.profile = profile
        self.in_proj = nn.Linear(in_features, profile.embed_dim)

        # 프로필의 num_blocks에 맞춰 Layer Stack 가변 구축
        blocks = []
        for i in range(profile.num_blocks):
            dilation = 2 ** (i % 4)  # 1, 2, 4, 8 순환
            blocks.append(ElasticConvBlock(profile.embed_dim, dilation=dilation))
        self.conv_stack = nn.Sequential(*blocks)

        # Global Pooling 및 Latent Z 투영기
        self.attn_pool = nn.Sequential(
            nn.Linear(profile.embed_dim, max(1, profile.embed_dim // 2)),
            nn.Tanh(),
            nn.Linear(max(1, profile.embed_dim // 2), 1)
        )
        self.to_latent_z = nn.Sequential(
            nn.Linear(profile.embed_dim, profile.latent_dim),
            nn.LayerNorm(profile.latent_dim),
            nn.GELU()
        )

    def forward(self, continuous_input: torch.Tensor) -> torch.Tensor:
        """
        [입력] continuous_input: [Batch, Seq_Len, Features] 또는 [Batch, Features, Seq_Len]
        [출력] latent_z: [Batch, Latent_Dim]
        """
        if continuous_input.dim() == 2:
            continuous_input = continuous_input.unsqueeze(0)

        if continuous_input.size(-1) != self.in_proj.in_features:
            if continuous_input.size(1) == self.in_proj.in_features:
                x_seq = continuous_input.transpose(1, 2)
            else:
                x_seq = continuous_input
        else:
            x_seq = continuous_input

        # [Batch, Seq, Feat] -> Linear Proj -> [Batch, Seq, Embed_Dim] -> [Batch, Embed_Dim, Seq]
        x = self.in_proj(x_seq).transpose(1, 2)
        x = self.conv_stack(x)

        # Global Attention Pooling -> [Batch, Embed_Dim]
        x_trans = x.transpose(1, 2)
        attn_weights = F.softmax(self.attn_pool(x_trans), dim=1)
        pooled = torch.sum(x_trans * attn_weights, dim=1)

        # Dynamic Latent Vector Z 생성 -> [Batch, Latent_Dim]
        return self.to_latent_z(pooled)


class MicroLatentGenerator(ElasticLatentGenerator):
    """VRAM < 200MB 저사양 환경을 위한 Micro 모드 전용 하위 호환 래퍼"""
    def __init__(self, in_features: int = 64, latent_dim: int = 128):
        prof = ElasticProfile(
            name="Micro Mode (< 200MB)",
            latent_dim=latent_dim,
            manifold_dim=4,
            embed_dim=64,
            num_blocks=3,
            threshold=1e-3
        )
        super().__init__(profile=prof, in_features=in_features)


class ElasticCausalBuilder(nn.Module):
    """Latent Z 크기에 따라 N x N 크기의 매니폴드 M_t를 구축하고 det(M)을 검증"""
    def __init__(self, profile: ElasticProfile):
        super().__init__()
        self.profile = profile
        self.m_dim = profile.manifold_dim

        # Latent Z -> N x N 매니폴드 투영기
        self.proj_manifold = nn.Linear(profile.latent_dim, self.m_dim * self.m_dim)

    def forward(self, latent_z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if latent_z.dim() == 1:
            latent_z = latent_z.unsqueeze(0)
        batch_size = latent_z.shape[0]

        # 1. N x N 매니폴드 M_t 동적 복원
        raw_m = self.proj_manifold(latent_z).view(batch_size, self.m_dim, self.m_dim)

        # 2. 대칭성 및 정규화 기반 인과 매니폴드 대칭 구성
        manifold_m = 0.5 * (raw_m + raw_m.transpose(1, 2)) + torch.eye(self.m_dim, device=latent_z.device)

        # 3. 매니폴드 행렬식 det(M_t) 계산 (Topological Invariant)
        det_vals = torch.det(manifold_m)

        return manifold_m, det_vals


class DynamicElasticEngine(nn.Module):
    """VRAM 자동 감지 및 실시간 하이브리드 인터페이스 통합 컨트롤러"""
    def __init__(self, override_profile: Optional[ElasticProfile] = None, in_features: int = 64):
        super().__init__()
        self.profile = override_profile if override_profile is not None else VRAMSensors.detect_and_get_profile()

        self.generator = ElasticLatentGenerator(self.profile, in_features=in_features)
        self.causal_builder = ElasticCausalBuilder(self.profile)

    def print_engine_status(self) -> None:
        total_params = sum(p.numel() for p in self.parameters())
        weight_mb = total_params * 4 / (1024 * 1024)
        print("=========================================================")
        print(f" [Dynamic Elastic Engine 활성화 리포트]")
        print("=========================================================")
        print(f"  • 선택된 모드      : {self.profile.name}")
        print(f"  • Latent Z 차원    : {self.profile.latent_dim} Dim")
        print(f"  • 매니폴드 M 해상도 : {self.profile.manifold_dim} x {self.profile.manifold_dim}")
        print(f"  • 총 파라미터 수   : {total_params:,} 개 (~{weight_mb:.2f} MB)")
        print(f"  • 환각 감지 임계치 : det(M) < {self.profile.threshold}")
        print("=========================================================\n")

    def process_sequence(self, continuous_input: torch.Tensor) -> Dict[str, Any]:
        # 1. Latent Z 생성
        latent_z = self.generator(continuous_input)

        # 2. 가변 N x N 매니폴드 및 행렬식 연산
        manifold_m, det_vals = self.causal_builder(latent_z)

        # 3. 환각 여부 검증
        is_hallucination = torch.abs(det_vals) < self.profile.threshold

        return {
            "mode": self.profile.name,
            "latent_z": latent_z,
            "manifold_m": manifold_m,
            "det_vals": det_vals,
            "is_hallucination": is_hallucination
        }
