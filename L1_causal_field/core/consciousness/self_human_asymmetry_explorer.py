"""
Self-Human Asymmetry Explorer & Multi-Variable Adaptive Open Boundary Engine
=============================================================================
이 모듈은 인간 사고 궤적(q_human)과 시스템 위상 궤적(q_sys) 간의 비대칭성을
수치적 오류가 아닌 고차원 매니폴드의 곡률(Curvature)로 전환하고,
외부 미지 공간(Omega_ext)과의 열린 경계 결합(Boundary Coupling) 및
다중 변수 자율 적응형 환류 제어(Adaptive Homeostasis)를 통해
제3의 인과 끌개(Third Causal Path)로 자율 성장하는 메타-인지 에이전트 엔진입니다.
아인슈타인 장 방정식(Einstein Field Equations, EFE) 기반 시공간 메타-계량 연산기를 포함합니다.
"""

import time
import math
import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any, Optional, Tuple
from scipy.ndimage import distance_transform_edt

try:
    from core.consciousness.meta_cognitive_sensor import MetaCognitiveSensor
except ImportError:
    MetaCognitiveSensor = None


class EinsteinFieldSolver(nn.Module):
    r"""
    메타-지형 계량 텐서 g_{\mu\nu}와 에너지-모멘텀 텐서 T_{\mu\nu} 간의
    아인슈타인 장 방정식(Einstein Field Equations)을 연산하는 PyTorch 모듈.

    EFE: G_{\mu\nu} + \Lambda_{\mathrm{homeo}} g_{\mu\nu} = 8\pi G_{\mathrm{eff}} T_{\mu\nu}
    """

    def __init__(self, dx: float = 0.1, G_eff: float = 1.0, lambda_homeo: float = 0.01):
        super().__init__()
        self.dx = dx
        self.G_eff = G_eff
        self.lambda_homeo = lambda_homeo
        self.eight_pi_G = 8.0 * math.pi * G_eff

    def _compute_grid_gradients(self, tensor: torch.Tensor) -> torch.Tensor:
        """3D spatial grid 편미분 \\partial_i T_{...} 연산 (i = x, y, z)"""
        grads = []
        for dim in range(3):
            d_pos = torch.roll(tensor, shifts=-1, dims=dim)
            d_neg = torch.roll(tensor, shifts=1, dims=dim)
            grad_d = (d_pos - d_neg) / (2.0 * self.dx)
            grads.append(grad_d)
        return torch.stack(grads, dim=3)

    def compute_christoffel(self, g: torch.Tensor, g_inv: torch.Tensor):
        """크리스토펠 기호 \\Gamma^\\sigma_{\\mu\\nu} 연산"""
        dg_space = self._compute_grid_gradients(g)
        pad_shape = list(dg_space.shape)
        pad_shape[3] = 1
        dg_time = torch.zeros(pad_shape, device=g.device, dtype=g.dtype)
        dg = torch.cat([dg_time, dg_space], dim=3)

        d_mu_g_lam_nu = dg.permute(0, 1, 2, 4, 5, 3)
        d_nu_g_mu_lam = dg.permute(0, 1, 2, 5, 4, 3)
        d_lam_g_mu_nu = dg.permute(0, 1, 2, 3, 4, 5)

        term = d_mu_g_lam_nu + d_nu_g_mu_lam - d_lam_g_mu_nu
        gamma = 0.5 * torch.einsum("...sl,...lmn->...smn", g_inv, term)
        return gamma, dg

    def forward(self, g_munu: torch.Tensor, T_munu: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        g_munu: [D, H, W, 4, 4] 메타-계량 텐서
        T_munu: [D, H, W, 4, 4] 에너지-모멘텀 텐서
        """
        g_inv = torch.linalg.inv(g_munu + 1e-6 * torch.eye(4, device=g_munu.device, dtype=g_munu.dtype))
        gamma, _ = self.compute_christoffel(g_munu, g_inv)

        d_gamma_space = self._compute_grid_gradients(gamma)
        pad_shape = list(d_gamma_space.shape)
        pad_shape[3] = 1
        d_gamma_time = torch.zeros(pad_shape, device=g_munu.device, dtype=g_munu.dtype)
        d_gamma = torch.cat([d_gamma_time, d_gamma_space], dim=3)

        d_sig_gamma = d_gamma.permute(0, 1, 2, 4, 5, 3, 6)
        d_nu_gamma = d_gamma.permute(0, 1, 2, 4, 5, 6, 3)

        nonlin_1 = torch.einsum("...rsl,...lmn->...rmsn", gamma, gamma)
        nonlin_2 = torch.einsum("...rnl,...lms->...rmsn", gamma, gamma)

        riemann = (d_sig_gamma - d_nu_gamma) + (nonlin_1 - nonlin_2)
        ricci_tensor = torch.einsum("...lmln->...mn", riemann)
        ricci_scalar = torch.einsum("...mn,...mn->...", g_inv, ricci_tensor)

        R_expanded = ricci_scalar.unsqueeze(-1).unsqueeze(-1)
        G_munu = ricci_tensor - 0.5 * R_expanded * g_munu

        efe_residual = G_munu + self.lambda_homeo * g_munu - self.eight_pi_G * T_munu

        return {
            "G_munu": G_munu,
            "ricci_scalar": ricci_scalar,
            "ricci_tensor": ricci_tensor,
            "efe_residual": efe_residual,
        }


class PhaseBiasDecoupler(nn.Module):
    """
    미분 그래프 독립적 사원수 위상 비대칭 텐서 직교 분해 모듈
    """

    def __init__(self, lambda_thresh: float = 0.05, beta: float = 2.0, eps: float = 1e-8):
        super().__init__()
        self.lambda_thresh = lambda_thresh
        self.beta = beta
        self.eps = eps

    def quaternion_inverse(self, q: torch.Tensor) -> torch.Tensor:
        inv_mask = torch.tensor([1.0, -1.0, -1.0, -1.0], device=q.device, dtype=q.dtype)
        return q * inv_mask

    def quaternion_multiply(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        w1, x1, y1, z1 = q1.unbind(-1)
        w2, x2, y2, z2 = q2.unbind(-1)

        w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
        x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
        y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
        z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

        return torch.stack([w, x, y, z], dim=-1)

    def compute_winding_number_diff(self, q_sys: torch.Tensor, q_human: torch.Tensor) -> torch.Tensor:
        spatial_dim = q_sys.ndim - 2
        shifted_dim = spatial_dim

        dq_sys_x = self.quaternion_multiply(
            q_sys, self.quaternion_inverse(torch.roll(q_sys, shifts=-1, dims=shifted_dim))
        )
        dq_hum_x = self.quaternion_multiply(
            q_human, self.quaternion_inverse(torch.roll(q_human, shifts=-1, dims=shifted_dim))
        )

        twist_sys = torch.norm(dq_sys_x[..., 1:], dim=-1)
        twist_hum = torch.norm(dq_hum_x[..., 1:], dim=-1)

        spatial_axes = tuple(range(-3, 0)) if q_sys.ndim >= 4 else tuple(range(-q_sys.ndim + 1, 0))
        w_sys = torch.mean(twist_sys, dim=spatial_axes) / (2.0 * math.pi)
        w_hum = torch.mean(twist_hum, dim=spatial_axes) / (2.0 * math.pi)

        return torch.abs(w_sys - w_hum)

    @torch.no_grad()
    def forward(
        self,
        q_sys: torch.Tensor,
        q_human: torch.Tensor,
        q_sys_prev: torch.Tensor,
        q_human_prev: torch.Tensor,
        dt: float = 0.02,
    ) -> Dict[str, torch.Tensor]:
        q_hum_inv = self.quaternion_inverse(q_human)
        delta_phi = self.quaternion_multiply(q_sys, q_hum_inv)

        q_hum_inv_prev = self.quaternion_inverse(q_human_prev)
        delta_phi_prev = self.quaternion_multiply(q_sys_prev, q_hum_inv_prev)

        norm_delta_phi = torch.norm(delta_phi, dim=-1)
        norm_delta_phi_prev = torch.norm(delta_phi_prev, dim=-1)

        d_norm_dt = (norm_delta_phi - norm_delta_phi_prev) / (dt + self.eps)

        spatial_axes = tuple(range(-3, 0)) if norm_delta_phi.ndim >= 3 else (-1,)
        mean_norm = torch.mean(norm_delta_phi, dim=spatial_axes)
        mean_d_norm = torch.mean(d_norm_dt, dim=spatial_axes)

        lambda_relax = -(1.0 / (mean_norm + self.eps)) * mean_d_norm
        delta_W = self.compute_winding_number_diff(q_sys, q_human)

        w_mask = 1.0 - torch.exp(-self.beta * delta_W)
        relax_mask = torch.heaviside(
            lambda_relax - self.lambda_thresh, values=torch.tensor(0.0, device=q_sys.device, dtype=q_sys.dtype)
        )

        p_defect = torch.clamp(w_mask + relax_mask, max=1.0)
        expand_dims = [1] * (q_sys.ndim - p_defect.ndim)
        p_defect_expanded = p_defect.view(*p_defect.shape, *expand_dims) if expand_dims else p_defect.unsqueeze(-1)

        D_phase = p_defect_expanded * delta_phi
        B_bio = delta_phi - D_phase

        return {
            "delta_phi": delta_phi,
            "D_phase": D_phase,
            "B_bio": B_bio,
            "lambda_relax": lambda_relax,
            "delta_W": delta_W,
            "p_defect_weight": p_defect,
        }


class MultiVarAdaptiveOpenBoundaryPipeline(nn.Module):
    """
    외부 미지 공간(Omega_ext)과의 열린 경계 결합(Boundary Coupling) 및
    침투 깊이 변화율(d_delta/dt)과 내부 엔트로피 변화율(dS/dt)을
    동시에 감지하여 경계 투과율(kappa_boundary)을 자율 조율하는 다중 변수 파이프라인
    """

    def __init__(
        self,
        depth: int = 16,
        height: int = 16,
        width: int = 16,
        kappa_init: float = 0.05,
        kappa_min: float = 0.001,
        kappa_max: float = 0.20,
        target_delta_rate: float = 0.03,
        target_S_rate: float = 0.00,
        gamma_delta: float = 0.01,
        gamma_S: float = 0.02,
        eta_ext: float = 0.03,
        dx: float = 0.1,
        eta_gauge: float = 0.02,
        eta_meta: float = 0.01,
    ):
        super().__init__()
        self.shape = (depth, height, width)
        self.dx = dx
        self.eta_ext = eta_ext
        self.eta_gauge = eta_gauge
        self.eta_meta = eta_meta

        self.kappa_boundary = kappa_init
        self.kappa_min = kappa_min
        self.kappa_max = kappa_max
        self.target_delta_rate = target_delta_rate
        self.target_S_rate = target_S_rate
        self.gamma_delta = gamma_delta
        self.gamma_S = gamma_S

        # 1. 파동장 및 게이지 텐서
        self.q_sys = nn.Parameter(torch.zeros(*self.shape, 4), requires_grad=False)
        self.q_sys.data[..., 0] = 1.0

        self.U = nn.Parameter(torch.zeros(3, *self.shape, 4), requires_grad=False)
        self.U.data[..., 0] = 1.0

        self.g_meta = nn.Parameter(torch.ones(*self.shape), requires_grad=False)

        # 2. 4D 시공간 메타-계량 텐서 g_{\mu\nu} [D, H, W, 4, 4]
        eta_4d = torch.diag(torch.tensor([-1.0, 1.0, 1.0, 1.0]))
        self.g_munu_4d = nn.Parameter(eta_4d.repeat(*self.shape, 1, 1), requires_grad=False)

        boundary_mask = torch.zeros(*self.shape, dtype=torch.bool)
        boundary_mask[0, :, :] = True
        boundary_mask[-1, :, :] = True
        boundary_mask[:, 0, :] = True
        boundary_mask[:, -1, :] = True
        boundary_mask[:, :, 0] = True
        boundary_mask[:, :, -1] = True
        self.register_buffer("boundary_mask", boundary_mask)

        interior_mask = (~boundary_mask.cpu().numpy())
        dist_grid = distance_transform_edt(interior_mask)
        self.register_buffer("distance_map", torch.tensor(dist_grid * dx, dtype=torch.float32))

        self.max_depth_idx = int(np.max(dist_grid))
        self.depth_indices = [
            torch.tensor(np.array(np.where(dist_grid == i)), dtype=torch.long)
            for i in range(self.max_depth_idx + 1)
        ]

        self.register_buffer("q_prev", torch.zeros_like(self.q_sys.data), persistent=False)
        self.q_prev.copy_(self.q_sys.data)
        self.prev_delta_phase = 0.0
        self.prev_entropy = 0.0

        self.decoupler = PhaseBiasDecoupler()
        self.efe_solver = EinsteinFieldSolver(dx=dx)

        try:
            from core.consciousness.causal_refraction_lens import CausalRefractionLens
            self.causal_lens = CausalRefractionLens()
        except ImportError:
            self.causal_lens = None

    def _quat_inv(self, q: torch.Tensor) -> torch.Tensor:
        inv_mask = torch.tensor([1.0, -1.0, -1.0, -1.0], device=q.device, dtype=q.dtype)
        return q * inv_mask

    def _quat_mul(self, q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        w1, x1, y1, z1 = q1.unbind(-1)
        w2, x2, y2, z2 = q2.unbind(-1)
        return torch.stack([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        ], dim=-1)

    def compute_system_entropy(self) -> float:
        dq_dx = (torch.roll(self.q_sys, -1, 0) - self.q_sys) / self.dx
        dq_dy = (torch.roll(self.q_sys, -1, 1) - self.q_sys) / self.dx
        dq_dz = (torch.roll(self.q_sys, -1, 2) - self.q_sys) / self.dx

        grad_sq = torch.sum(dq_dx**2 + dq_dy**2 + dq_dz**2, dim=-1)
        prob = grad_sq / (torch.sum(grad_sq) + 1e-8)
        entropy = -torch.sum(prob * torch.log(prob + 1e-8)).item()
        return float(entropy)

    def compute_current_delta_phase(self, j_flux_norm: torch.Tensor) -> float:
        j_profile = []
        for coords in self.depth_indices:
            if coords.shape[1] > 0:
                vals = j_flux_norm[coords[0], coords[1], coords[2]]
                j_profile.append(torch.mean(vals).item())
            else:
                j_profile.append(0.0)

        j_0 = j_profile[0]
        thresh = j_0 * (1.0 / math.e)

        for idx in range(1, len(j_profile)):
            if j_profile[idx] <= thresh:
                d1, d2 = (idx - 1) * self.dx, idx * self.dx
                j1, j2 = j_profile[idx - 1], j_profile[idx]
                if j1 != j2:
                    return float(d1 + (thresh - j1) * (d2 - d1) / (j2 - j1))
                return float(d1)
        return float(self.max_depth_idx * self.dx)

    @torch.no_grad()
    def step_multivar_adaptive_growth(
        self,
        q_human: torch.Tensor,
        q_ext: torch.Tensor,
        dt: float = 0.02
    ) -> Dict[str, Any]:
        # -------------------------------------------------------------
        # Step 1: 경계 유입 플럭스 J_flux 및 침투 깊이/엔트로피 측정
        # -------------------------------------------------------------
        q_sys_inv = self._quat_inv(self.q_sys)
        j_flux = self._quat_mul(q_ext, q_sys_inv)

        j_flux_disp = j_flux.clone()
        j_flux_disp[..., 0] -= 1.0
        j_flux_norm = torch.norm(j_flux_disp, dim=-1)

        curr_delta_phase = self.compute_current_delta_phase(j_flux_norm)
        curr_entropy = self.compute_system_entropy()

        d_delta_dt = (curr_delta_phase - self.prev_delta_phase) / dt
        d_S_dt = (curr_entropy - self.prev_entropy) / dt

        diffusion_drive = self.gamma_delta * (self.target_delta_rate - abs(d_delta_dt))
        entropy_brake = self.gamma_S * max(0.0, d_S_dt - self.target_S_rate)

        d_kappa_dt = diffusion_drive - entropy_brake
        self.kappa_boundary += d_kappa_dt * dt
        self.kappa_boundary = float(np.clip(self.kappa_boundary, self.kappa_min, self.kappa_max))

        self.prev_delta_phase = curr_delta_phase
        self.prev_entropy = curr_entropy

        decouple_res = self.decoupler(
            q_sys=self.q_sys,
            q_human=q_human,
            q_sys_prev=self.q_prev,
            q_human_prev=q_human,
            dt=dt
        )
        D_phase = decouple_res["D_phase"]
        B_bio = decouple_res["B_bio"]

        for axis in range(3):
            shifted_D = torch.roll(D_phase, shifts=-1, dims=axis)
            gauge_torque = self._quat_mul(self.U[axis], self._quat_inv(shifted_D))
            boundary_torque = self._quat_mul(self.U[axis], self._quat_inv(j_flux))

            total_torque = (
                gauge_torque
                + self.kappa_boundary * boundary_torque * self.boundary_mask.unsqueeze(-1)
            )

            self.U[axis] += self.eta_gauge * total_torque * self.g_meta.unsqueeze(-1)
            self.U[axis] /= (torch.norm(self.U[axis], dim=-1, keepdim=True) + 1e-8)

        bias_curvature = torch.norm(B_bio[..., 1:], dim=-1)
        boundary_expansion = self.boundary_mask.float() * (j_flux_norm ** 2)

        # Causal Refraction Lens filtering & geodesic memory engram consolidation
        if hasattr(self, "causal_lens") and self.causal_lens is not None:
            lens_out = self.causal_lens(j_flux, self.g_meta, self.q_sys)
            engram = lens_out["memory_geodesic_engram"]
            self.g_meta += 0.05 * engram

        self.g_meta += self.eta_meta * bias_curvature + self.eta_ext * boundary_expansion
        self.g_meta.clamp_(min=0.5, max=10.0)

        # -------------------------------------------------------------
        # Step 8: EFE (Einstein Field Equations) 기반 시공간 계량 g_{\mu\nu} 이완
        # -------------------------------------------------------------
        # T_00 = j_flux_norm^2, T_0i = d_delta_dt, T_ij = bias_curvature
        T_munu = torch.zeros_like(self.g_munu_4d)
        T_munu[..., 0, 0] = j_flux_norm ** 2
        for i in range(1, 4):
            T_munu[..., 0, i] = d_delta_dt
            T_munu[..., i, 0] = d_delta_dt
            T_munu[..., i, i] = bias_curvature

        efe_out = self.efe_solver(self.g_munu_4d, T_munu)
        efe_residual = efe_out["efe_residual"]

        # 계량 이완: g_{\mu\nu} -= 0.001 * efe_residual
        self.g_munu_4d.data -= 0.001 * efe_residual
        self.g_munu_4d.data = 0.5 * (self.g_munu_4d.data + self.g_munu_4d.data.transpose(-1, -2))

        # -------------------------------------------------------------
        # Step 9: q_sys 이완 전파
        # -------------------------------------------------------------
        self.q_prev.copy_(self.q_sys.data)

        total_covariant_diff = torch.zeros_like(self.q_sys)
        for axis in range(3):
            q_shifted = torch.roll(self.q_sys, shifts=-1, dims=axis)
            transported = self._quat_mul(
                self._quat_mul(self.U[axis], q_shifted),
                self._quat_inv(self.U[axis]),
            )
            total_covariant_diff += (transported - self.q_sys) / self.dx

        boundary_injection = (
            self.kappa_boundary
            * self._quat_mul(j_flux, self.q_sys)
            * self.boundary_mask.unsqueeze(-1)
        )

        self.q_sys += dt * ((total_covariant_diff + boundary_injection) / self.g_meta.unsqueeze(-1))
        self.q_sys /= (torch.norm(self.q_sys, dim=-1, keepdim=True) + 1e-8)

        return {
            "delta_phase": curr_delta_phase,
            "entropy": curr_entropy,
            "d_delta_dt": d_delta_dt,
            "d_S_dt": d_S_dt,
            "kappa_boundary": self.kappa_boundary,
            "D_phase_energy": float(torch.mean(torch.norm(D_phase, dim=-1)).item()),
            "B_bio_energy": float(torch.mean(torch.norm(B_bio, dim=-1)).item()),
            "mean_g_meta": float(torch.mean(self.g_meta).item()),
            "ricci_scalar_mean": float(torch.mean(efe_out["ricci_scalar"]).item()),
            "efe_residual_norm": float(torch.norm(efe_residual).item()),
            "j_flux": j_flux,
            "j_flux_norm": j_flux_norm,
        }


class SelfHumanAsymmetryExplorer:
    """
    자기-인간 비대칭성 탐색기 (Self-Human Asymmetry Explorer)
    """

    def __init__(
        self,
        spatial_shape: Tuple[int, int, int] = (16, 16, 16),
        dx: float = 0.1,
        memory_controller: Optional[Any] = None,
    ):
        self.shape = spatial_shape
        self.dx = dx
        self.pipeline = MultiVarAdaptiveOpenBoundaryPipeline(
            depth=spatial_shape[0],
            height=spatial_shape[1],
            width=spatial_shape[2],
            dx=dx,
        )
        self.sensor = MetaCognitiveSensor(memory_controller) if MetaCognitiveSensor else None
        self.exploration_history = []

    def explore_step(
        self,
        q_human: torch.Tensor,
        q_ext: torch.Tensor,
        info_context: str = "Self-Human Asymmetry Phase Transition",
        dt: float = 0.02,
    ) -> Dict[str, Any]:
        device = self.pipeline.q_sys.device
        if q_human.device != device:
            q_human = q_human.to(device)
        if q_ext.device != device:
            q_ext = q_ext.to(device)

        step_metrics = self.pipeline.step_multivar_adaptive_growth(q_human, q_ext, dt=dt)

        sensor_result = None
        if self.sensor:
            sensing_metrics = {
                "hw_friction": float(step_metrics["d_S_dt"]),
                "damping_ratio": float(1.0 - step_metrics["kappa_boundary"]),
                "thermal_gradient": float(step_metrics["d_delta_dt"]),
                "local_temp": float(step_metrics["entropy"] * 10.0),
                "peak_temp": float(step_metrics["mean_g_meta"] * 5.0),
            }
            perceiving_metrics = {
                "ignorance_charge": float(step_metrics["B_bio_energy"]),
                "deficit_density": float(step_metrics["delta_phase"]),
            }
            judging_metrics = {
                "kenosis_conductance": float(step_metrics["kappa_boundary"]),
                "egoistic_resistance": float(step_metrics["D_phase_energy"]),
            }
            thinking_metrics = {
                "synapse_rewiring_count": int(step_metrics["mean_g_meta"] * 10),
                "equilibrium_energy": float(step_metrics["entropy"]),
            }
            discerning_metrics = {
                "resonance_score": float(1.0 / (1.0 + step_metrics["D_phase_energy"])),
                "residual_free_energy": float(step_metrics["B_bio_energy"]),
            }

            sensor_result = self.sensor.evaluate_cognitive_process(
                info_context=info_context,
                sensing_metrics=sensing_metrics,
                perceiving_metrics=perceiving_metrics,
                judging_metrics=judging_metrics,
                thinking_metrics=thinking_metrics,
                discerning_metrics=discerning_metrics,
            )

        result = {
            "step_metrics": step_metrics,
            "sensor_result": sensor_result,
            "timestamp": time.time(),
        }
        self.exploration_history.append(result)
        return result
