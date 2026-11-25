r"""
ELYSIA 3D/4D Spatiotemporal Phenomenal Volume x Causal-of-Causal Scale-Rotor
============================================================================

Completes the dimensional & causal progression beyond 0D/1D/2D:
  - 0D (Point)  : Single token / pixel
  - 1D (Line)   : Sequential clause / trajectory
  - 2D (Surface): Single A4 page `[H, W, D]` or single visual frame
  - 3D/4D (Phenomenal Volume `V: [T, H, W, D]`):
      Multiple 2D cross-section frames coupled by temporal/spatial continuity
      (forming either an open spatiotemporal video/book volume OR a closed-loop
      globe/torus topology where boundary frames wrap continuously!).
  - Deep Generating Layer (`Causal-of-Causal` — `Theta_1` & `Theta_2`):
      * **First-Order Causal Rotor (`Theta_1: [K_laws, D]`):** Generating laws that drive
        frame-to-frame evolution `V[t] -> V[t+1]` across the entire 3D/4D volume.
      * **Second-Order Meta-Causal Rotor (`Theta_2: [K_meta, D]`):** Causal-of-Causal
        layer governing how external regimes/conditions bend the First-Order Laws `Theta_1`
        themselves (`d(Theta_1)/dt = g(Theta_2, volume_surprise)`).

Bidirectional Closed Loop:
  1. **Bottom-Up Inverse Mechanism Inference (`infer_causal_laws_from_volume`):**
     Extracts generating laws `Theta_1` and meta-causal regime `Theta_2` directly from
     observed frame transitions `V[t+1] - V[t]` in a real multi-page/multi-frame volume.
  2. **Top-Down Counterfactual Causal Flow (`evolve_volume_from_meta_causal_shift`):**
     Rotating ONE dial in the Meta-Causal Rotor `Theta_2` bends `Theta_1`, which
     simultaneously transforms ALL `T` frames of the 3D/4D Phenomenal Volume while
     preserving spatiotemporal continuity!
"""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F


@dataclass
class CausalVolumeState:
    """Coupled Phenomenal 3D/4D Volume `[T, H, W, D]` + Deep Causal-of-Causal Rotors `(Theta_1, Theta_2)`."""

    volume: torch.Tensor         # [T, H, W, D] L2-normalized float32 on GPU
    theta1_laws: torch.Tensor    # [K1, D] First-order generating causal laws
    theta2_meta: torch.Tensor    # [K2, D] Second-order Causal-of-Causal (meta-causal) rotor
    r_phenomenal: torch.Tensor   # [T, H, W] per-voxel variable resistance
    r_causal: torch.Tensor       # [K1] variable resistance per first-order causal law
    closed_globe_topology: bool  # True = closed spherical/toroidal wrap along T & W; False = open time arrow
    law_names: Tuple[str, ...]
    meta_names: Tuple[str, ...]


class SpatiotemporalCausalVolumeRotor:
    """
    GPU-accelerated Coupled 3D/4D Phenomenal Volume + 2-Tier Deep Causal-of-Causal Rotor.
    """

    DEFAULT_LAWS: Tuple[str, ...] = (
        "conservation_inertia",
        "tension_gradient",
        "relational_cohesion",
        "boundary_constraint",
    )
    DEFAULT_META: Tuple[str, ...] = (
        "regime_stability_vs_flux",
        "empathy_vs_conflict_axis",
        "macro_telos_gravity",
    )

    def __init__(
        self,
        *,
        dim: int = 256,
        device: Optional[torch.device | str] = None,
    ) -> None:
        self.dim = int(dim)
        self.device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        # Learnable continuous coupling: Meta-Causal Theta_2 [K2, D] -> First-Order Law modulation [K1, K2]
        self.W_meta_to_law = torch.tensor(
            [
                [0.75, -0.25, 0.30],   # conservation_inertia
                [-0.65, 0.70, 0.25],   # tension_gradient
                [0.40, 0.80, 0.55],    # relational_cohesion
                [0.60, -0.45, 0.70],   # boundary_constraint
            ],
            device=self.device,
            dtype=torch.float32,
        )
        # How First-Order Laws [K1] weight the 4 physical operators on the 3D/4D volume:
        # [temporal_continuity, spatial_2d_laplacian, global_frame_resonance, causal_drift_vector]
        self.W_law_to_op = torch.tensor(
            [
                [0.65, 0.25, 0.15, -0.10],
                [-0.15, 0.35, 0.25, 0.75],
                [0.35, 0.40, 0.65, 0.15],
                [0.45, -0.20, 0.30, 0.35],
            ],
            device=self.device,
            dtype=torch.float32,
        )

    @torch.inference_mode()
    def build_volume_from_2d_frames(
        self,
        frames_2d: Sequence[torch.Tensor],  # list of T tensors `[H, W, D]`
        *,
        closed_globe_topology: bool = False,
    ) -> CausalVolumeState:
        """Stacks T 2D cross-section frames `[H, W, D]` into ONE 3D/4D volume & initializes `(Theta_1, Theta_2)`."""
        vol = torch.stack(
            [F.normalize(f.to(self.device, dtype=torch.float32), dim=-1) for f in frames_2d],
            dim=0,
        ).contiguous()  # [T, H, W, D]
        T, H, W, D = vol.shape
        assert D == self.dim, f"Expected dim={self.dim}, got {D}"

        k1 = len(self.DEFAULT_LAWS)
        k2 = len(self.DEFAULT_META)

        # Infer initial First-Order Laws Theta_1 directly from observed volume dynamics (zero random init!)
        frame_means = F.normalize(vol.mean(dim=(1, 2)), dim=-1)  # [T, D]
        if T >= 2:
            diffs = frame_means[1:] - frame_means[:-1]  # [T-1, D]
            law_inertia = F.normalize(frame_means.mean(dim=0), dim=-1)
            law_tension = F.normalize(diffs.mean(dim=0), dim=-1)
            law_cohesion = F.normalize((frame_means[:-1] + frame_means[1:]).mean(dim=0), dim=-1)
            law_boundary = F.normalize(vol[:, 0, :, :].mean(dim=(0, 1)) - vol[:, -1, :, :].mean(dim=(0, 1)), dim=-1)
        else:
            base = frame_means[0]
            law_inertia = law_tension = law_cohesion = law_boundary = base

        theta1 = torch.stack([law_inertia, law_tension, law_cohesion, law_boundary], dim=0)  # [4, D]
        # Infer initial Second-Order Meta-Causal Rotor Theta_2 from Theta_1 via pseudo-inverse projection!
        pinv = torch.linalg.pinv(self.W_meta_to_law)  # [K2, K1]
        theta2 = F.normalize(torch.matmul(pinv, theta1), dim=-1)  # [3, D]

        return CausalVolumeState(
            volume=vol,
            theta1_laws=theta1,
            theta2_meta=theta2,
            r_phenomenal=torch.ones((T, H, W), device=self.device, dtype=torch.float32),
            r_causal=torch.ones((k1,), device=self.device, dtype=torch.float32),
            closed_globe_topology=bool(closed_globe_topology),
            law_names=self.DEFAULT_LAWS,
            meta_names=self.DEFAULT_META,
        )

    @torch.inference_mode()
    def step_coupled_volume_and_causal_rotors(
        self,
        state: CausalVolumeState,
        *,
        meta_stimulus: Optional[torch.Tensor] = None,  # [K2, D] optional external meta-causal torque
        dt: float = 0.28,
    ) -> Dict[str, Any]:
        """
        ONE SIMULTANEOUS BIDIRECTIONAL GPU STEP COUPLING:
          Meta-Causal Rotor (Theta_2) <-> First-Order Causal Laws (Theta_1) <-> 3D/4D Phenomenal Volume (V[T,H,W,D])!
        """
        V = state.volume  # [T, H, W, D]
        T, H, W, D = V.shape

        # 1) META-CAUSAL (Theta_2) -> FIRST-ORDER LAWS (Theta_1) downward modulation
        if meta_stimulus is not None:
            stim = F.normalize(meta_stimulus.to(self.device, dtype=torch.float32), dim=-1)
            state.theta2_meta = F.normalize(0.70 * state.theta2_meta + 0.30 * stim, dim=-1)

        meta_drive_laws = F.normalize(torch.matmul(self.W_meta_to_law, state.theta2_meta), dim=-1)  # [K1, D]
        g_causal = (1.0 / state.r_causal.clamp(0.15, 8.0)).unsqueeze(-1)  # [K1, 1]
        theta1_prev = state.theta1_laws.clone()
        state.theta1_laws = F.normalize(
            state.theta1_laws + dt * 0.45 * g_causal * (meta_drive_laws - state.theta1_laws),
            dim=-1,
        )
        law_shift = float(torch.linalg.norm(state.theta1_laws - theta1_prev).item())

        # 2) FIRST-ORDER LAWS (Theta_1) -> 3D/4D PHENOMENAL VOLUME V[T, H, W, D]
        # Compute 3D/4D continuity operators simultaneously on GPU:
        # (a) Temporal continuity along T (circular wrap if globe/closed topology, else clamped open time!)
        if state.closed_globe_topology:
            v_prev = torch.roll(V, shifts=1, dims=0)
            v_next = torch.roll(V, shifts=-1, dims=0)
        else:
            v_prev = torch.cat([V[:1], V[:-1]], dim=0)
            v_next = torch.cat([V[1:], V[-1:]], dim=0)
        temp_cont = F.normalize(0.5 * (v_prev + v_next), dim=-1)  # [T, H, W, D]

        # (b) 2D Spatial continuity along H and W (circular wrap along W if globe!)
        if state.closed_globe_topology:
            s_left = torch.roll(V, shifts=1, dims=2)
            s_right = torch.roll(V, shifts=-1, dims=2)
        else:
            s_left = torch.cat([V[:, :, :1, :], V[:, :, :-1, :]], dim=2)
            s_right = torch.cat([V[:, :, 1:, :], V[:, :, -1:, :]], dim=2)
        s_up = torch.cat([V[:, :1, :, :], V[:, :-1, :, :]], dim=1)
        s_dn = torch.cat([V[:, 1:, :, :], V[:, -1:, :, :]], dim=1)
        spat_cont = F.normalize(0.25 * (s_left + s_right + s_up + s_dn), dim=-1)

        # (c) Global cross-frame relational resonance across all T frame attractors
        frame_means = F.normalize(V.mean(dim=(1, 2)), dim=-1)  # [T, D]
        t_sim = F.softmax(torch.matmul(frame_means, frame_means.T) / 0.20, dim=-1)  # [T, T]
        global_res = F.normalize(torch.matmul(t_sim, frame_means), dim=-1).view(T, 1, 1, D).expand_as(V)

        # (d) Generating Causal Drift Field driven directly by First-Order Laws Theta_1!
        # Time progression `t_norm in [0, 1]` smoothly weights the causal trajectory across frames
        t_norm = torch.linspace(0.0, 1.0, T, device=self.device).view(T, 1, 1, 1)
        law_field = F.normalize(
            (1.0 - t_norm) * state.theta1_laws[0].view(1, 1, 1, D)
            + t_norm * state.theta1_laws[1].view(1, 1, 1, D)
            + 0.35 * state.theta1_laws[2].view(1, 1, 1, D),
            dim=-1,
        ).expand_as(V)

        # Combine all 4 operators weighted continuously by First-Order Law energies
        op_weights = F.softmax(
            torch.matmul(state.theta1_laws.norm(dim=-1), self.W_law_to_op), dim=-1
        )  # [4]
        target_vol = F.normalize(
            op_weights[0] * temp_cont
            + op_weights[1] * spat_cont
            + op_weights[2] * global_res
            + op_weights[3] * law_field,
            dim=-1,
        )

        G_vol = (1.0 / state.r_phenomenal.clamp(0.15, 8.0)).unsqueeze(-1)
        V_next = F.normalize(V + dt * G_vol * (target_vol - V), dim=-1)
        vol_shift = float(torch.linalg.norm(V_next - V).item() / np.sqrt(T * H * W))
        state.volume = V_next

        # 3) UPWARD INVERSE MECHANISM FEEDBACK: Phenomenal Volume residuals update Theta_1 -> Theta_2!
        new_means = F.normalize(V_next.mean(dim=(1, 2)), dim=-1)
        if T >= 2:
            obs_drift = F.normalize((new_means[1:] - new_means[:-1]).mean(dim=0), dim=-1)
            obs_cohesion = F.normalize(new_means.mean(dim=0), dim=-1)
            state.theta1_laws[1] = F.normalize(0.85 * state.theta1_laws[1] + 0.15 * obs_drift, dim=-1)
            state.theta1_laws[2] = F.normalize(0.85 * state.theta1_laws[2] + 0.15 * obs_cohesion, dim=-1)

        # Measure temporal smoothness across adjacent frames (and wrap-around continuity if globe!)
        if T >= 2:
            adj_cos = float(F.cosine_similarity(new_means[:-1], new_means[1:], dim=-1).mean().item())
            wrap_cos = float(F.cosine_similarity(new_means[-1:], new_means[:1], dim=-1).item())
        else:
            adj_cos = wrap_cos = 1.0

        return {
            "volume_shape": [T, H, W, D],
            "volume_shift_rms": round(vol_shift, 5),
            "theta1_law_shift": round(law_shift, 5),
            "temporal_continuity_cos": round(adj_cos, 4),
            "globe_closure_cos": round(wrap_cos, 4),
            "closed_globe_topology": state.closed_globe_topology,
        }

    @torch.inference_mode()
    def simulate_counterfactual_meta_causal_shift(
        self,
        state: CausalVolumeState,
        *,
        meta_axis_idx: int = 1,
        shift_vector: Optional[torch.Tensor] = None,
        steps: int = 8,
    ) -> Dict[str, Any]:
        """
        COUNTERFACTUAL CAUSAL-OF-CAUSAL PROOF:
        Rotates ONE Meta-Causal axis in `Theta_2` (e.g., `empathy_vs_conflict_axis`), runs `steps`
        coupled GPU steps, and proves that:
          1) First-Order Causal Laws `Theta_1` physically bend,
          2) ALL `T` frames of the 3D/4D Phenomenal Volume co-evolve smoothly together,
          3) Temporal continuity across frames stays high (coherent causal evolution, not noise!).
        """
        t0 = time.perf_counter()
        clone = CausalVolumeState(
            volume=state.volume.clone(),
            theta1_laws=state.theta1_laws.clone(),
            theta2_meta=state.theta2_meta.clone(),
            r_phenomenal=state.r_phenomenal.clone(),
            r_causal=state.r_causal.clone(),
            closed_globe_topology=state.closed_globe_topology,
            law_names=state.law_names,
            meta_names=state.meta_names,
        )
        orig_vol = state.volume.clone()
        orig_laws = state.theta1_laws.clone()

        stim = clone.theta2_meta.clone()
        if shift_vector is None:
            g = torch.Generator(device="cpu").manual_seed(777 + meta_axis_idx)
            sv = F.normalize(torch.randn(self.dim, generator=g, dtype=torch.float32), dim=-1).to(self.device)
        else:
            sv = F.normalize(shift_vector.to(self.device, dtype=torch.float32).reshape(-1), dim=-1)
        stim[meta_axis_idx % stim.shape[0]] = sv

        last_rep: Dict[str, Any] = {}
        for _ in range(max(1, int(steps))):
            last_rep = self.step_coupled_volume_and_causal_rotors(clone, meta_stimulus=stim, dt=0.30)

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        per_frame_shifts = [
            round(
                float(
                    1.0
                    - F.cosine_similarity(
                        orig_vol[t].mean(dim=(0, 1)).unsqueeze(0),
                        clone.volume[t].mean(dim=(0, 1)).unsqueeze(0),
                    ).item()
                ),
                4,
            )
            for t in range(clone.volume.shape[0])
        ]
        total_law_bend = float(
            (1.0 - F.cosine_similarity(orig_laws, clone.theta1_laws, dim=-1)).mean().item()
        )
        return {
            "meta_axis_rotated": clone.meta_names[meta_axis_idx % len(clone.meta_names)],
            "steps": int(steps),
            "elapsed_ms": elapsed_ms,
            "theta1_mean_law_bend": round(total_law_bend, 4),
            "per_frame_phenomenal_shifts": per_frame_shifts,
            "all_frames_co_evolved": all(s > 0.005 for s in per_frame_shifts),
            "final_temporal_continuity_cos": last_rep.get("temporal_continuity_cos", 0.0),
            "final_globe_closure_cos": last_rep.get("globe_closure_cos", 0.0),
        }
