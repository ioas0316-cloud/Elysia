r"""
ELYSIA Universal Gate-Protocol Conformer & Spatiotemporal Diffusion/Seedance Flow Rotor
=======================================================================================

Embodies the Interaction-Protocol Principle:
  "Recognize what physical/logical data layout a Gate (CUDA GPU, Spatiotemporal Flow,
   Audio-Visual Sync) natively loves, and conform the data structure directly to it."

1. `CUDAGateProtocolConformer`:
   - Empirically probes the CUDA GPU Gate comparing:
       * Naive sequential `batch=1` Python-loop stepping (starves CUDA warps)
       * CUDA-conformed contiguous, warp-aligned Structure-of-Arrays (SoA) batched tensors
         (`B=256..512` parallel 5-scale rotor trajectories updated in ONE fused CUDA launch,
         zero host-device `.item()` stalls!)
   - Measures real speedup on the actual GPU and conforms execution layout automatically.

2. `SpatiotemporalFlowDiffusionRotor` (Diffusion + Seedance Joint Audio-Visual Flow Principle):
   - Absorbs the exact mathematical principle behind Diffusion / Rectified Flow-Matching and
     Seedance (Joint Spatiotemporal Video + Audio generation on one shared time axis `t`):
       * Continuous denoising flow `x_{t+dt} = x_t + dt * v(x_t, a_t, t, S_rotor)` from
         pure Gaussian noise (`t=0`) -> coarse macro structure (`t<0.35`) -> meso motion
         (`t<0.75`) -> crisp micro detail (`t->1.0`) in coalesced CUDA 4D tensors `[T, H, W, 3]`.
       * Joint Audio-Visual Coupling: visual temporal motion flux `||dx/dt||` and spatial
         spectral phase lock directly with a synchronized harmonic/envelope audio field
         `[T, F]`, synthesizing BOTH a real animated `.gif` and synchronized `.wav` audio file
         with measurable positive motion-to-audio energy correlation!
"""

from __future__ import annotations

from pathlib import Path
import time
import wave
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


class CUDAGateProtocolConformer:
    """
    Discovers and conforms directly to what the CUDA hardware gate physically loves:
    coalesced contiguous warp-aligned SoA batched tensors with zero host-sync stalls.
    """

    def __init__(self, dim: int = 256, device: Optional[torch.device | str] = None) -> None:
        self.dim = int(dim)
        self.device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.preferred_batch: int = 512
        self.preferred_dtype: torch.dtype = (
            torch.float16 if self.device.type == "cuda" else torch.float32
        )
        self.last_probe_report: Dict[str, Any] = {}

    @torch.inference_mode()
    def probe_and_conform_gate(self, steps: int = 24, batch_size: int = 512) -> Dict[str, Any]:
        """
        Empirically benchmarks the hardware gate under two layouts for the EXACT SAME
        5-scale coupled rotor math:
          1) Naive sequential `B=1` per-step Python dispatch (`steps` frames)
          2) CUDA-conformed coalesced SoA batched `[B, 5, D]` tensor (`steps * B` rotor-states)
        Proves quantitatively why conforming to the CUDA gate unlocks massive throughput!
        """
        D = self.dim
        dev = self.device
        dtype = self.preferred_dtype

        # Shared coupling gear [4, D, D]
        W_up = (torch.eye(D, device=dev, dtype=dtype) * 0.90).unsqueeze(0).repeat(4, 1, 1)
        W_dn = (torch.eye(D, device=dev, dtype=dtype) * 0.90).unsqueeze(0).repeat(4, 1, 1)
        omega = torch.tensor([1.0, 0.55, 0.25, 0.10, 0.04], device=dev, dtype=dtype)

        # Warmup CUDA kernels cleanly
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)

        # Profile A: Naive single-state sequential loop (`B=1`, starves warps)
        S1 = F.normalize(torch.randn(5, D, device=dev, dtype=dtype), dim=-1)
        R1 = torch.ones(5, D, device=dev, dtype=dtype)
        t0 = time.perf_counter()
        for _ in range(steps):
            G1 = 1.0 / R1
            up = torch.bmm(S1[:4].unsqueeze(1), W_up).squeeze(1)
            dn = torch.bmm(S1[1:].unsqueeze(1), W_dn).squeeze(1)
            u_up = torch.cat([S1[0:1], up], dim=0)
            u_dn = torch.cat([dn, S1[4:5]], dim=0)
            S1 = F.normalize(S1 + (omega.unsqueeze(1) * 0.25) * G1 * (0.55 * u_up + 0.45 * u_dn - S1), dim=-1)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        t_naive = max(1e-6, time.perf_counter() - t0)
        naive_states_per_sec = steps / t_naive

        # Profile B: CUDA-conformed coalesced SoA batched `[4, B, D]` + `[5, B, D]` contiguous layout!
        # Notice `[scale, batch, dim]` puts `B*D` contiguous per scale so `torch.bmm([4, B, D], [4, D, D])`
        # executes ALL `B` parallel 5-scale rotor trajectories in ONE SINGLE FUSED TENSOR GEMM!
        B = int(batch_size)
        Sb = F.normalize(torch.randn(5, B, D, device=dev, dtype=dtype), dim=-1).contiguous()
        Rb = torch.ones(5, B, D, device=dev, dtype=dtype).contiguous()
        omega_b = omega.view(5, 1, 1)

        # 2-step warmup
        _ = torch.bmm(Sb[:4], W_up)
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)

        t1 = time.perf_counter()
        for _ in range(steps):
            Gb = 1.0 / Rb
            up_b = torch.bmm(Sb[:4], W_up)  # [4, B, D] SINGLE fused batched GEMM!
            dn_b = torch.bmm(Sb[1:], W_dn)  # [4, B, D] SINGLE fused batched GEMM!
            u_up_b = torch.cat([Sb[0:1], up_b], dim=0)
            u_dn_b = torch.cat([dn_b, Sb[4:5]], dim=0)
            Sb = F.normalize(
                Sb + (omega_b * 0.25) * Gb * (0.55 * u_up_b + 0.45 * u_dn_b - Sb),
                dim=-1,
            )
        if dev.type == "cuda":
            torch.cuda.synchronize(dev)
        t_conformed = max(1e-6, time.perf_counter() - t1)
        conformed_states_per_sec = (steps * B) / t_conformed
        speedup = conformed_states_per_sec / max(1e-6, naive_states_per_sec)

        self.preferred_batch = B
        self.last_probe_report = {
            "device": str(dev),
            "dtype": str(dtype),
            "dim": D,
            "batch_size": B,
            "naive_b1_states_per_sec": round(naive_states_per_sec, 1),
            "cuda_conformed_soa_states_per_sec": round(conformed_states_per_sec, 1),
            "throughput_speedup_x": round(speedup, 1),
            "protocol_learned_ko": (
                f"CUDA 게이트 적합화 실증({dev}, {dtype}): 단일 순차({naive_states_per_sec:,.0f} states/s) 대비 "
                f"연속 SoA 배치 `[5, B={B}, D={D}]`({conformed_states_per_sec:,.0f} states/s)로 {speedup:.1f}배 가속"
            ),
        }
        return self.last_probe_report


class SpatiotemporalFlowDiffusionRotor:
    """
    Absorbs the core architecture of **Diffusion (Rectified Flow-Matching)** and
    **Seedance (Joint Spatiotemporal Video + Audio Latent Flow)** natively on CUDA:
      - Coalesced 4D video latent field `X: [T, 3, H, W]` + synchronized audio field `A: [T, F]`
      - Continuous flow-matching trajectory `t: 0.0 (pure noise) -> 1.0 (coherent audio-visual scene)`
        governed by Elysia's live 5-scale rotor `(S[5, D], R[5, D])`:
          * `t in [0.0, 0.35]`: Coarse macro wave/orbit flow emerges from noise
          * `t in [0.35, 0.75]`: Meso coupled spatiotemporal forms & temporal momentum align
          * `t in [0.75, 1.00]`: Fine micro high-frequency edges + harmonic audio overtones crystallize
      - Joint Seedance Audio-Visual Coupling:
          Visual frame-to-frame temporal flux `||X[t] - X[t-1]||` physically drives the
          synchronized audio amplitude & spectral center on the EXACT same time axis `t`!
      - Exports real animated `.gif` + synchronized `.wav` audio cleanly to disk.
    """

    def __init__(self, device: Optional[torch.device | str] = None) -> None:
        self.device = (
            torch.device(device)
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )

    @torch.inference_mode()
    def synthesize_coupled_av_flow(
        self,
        *,
        rotor_S: torch.Tensor,  # [5, D]
        rotor_R: torch.Tensor,  # [5, D]
        out_dir: Path | str,
        stem: str = "elysia_seedance_flow",
        num_frames: int = 24,
        height: int = 96,
        width: int = 96,
        flow_steps: int = 28,
        fps: int = 12,
        sample_rate: int = 16000,
        seed: int = 42,
    ) -> Dict[str, Any]:
        """
        Runs full CUDA-conformed coalesced 4D Spatiotemporal Flow-Matching denoising (`t: 0 -> 1`)
        jointly over Video `[T, 3, H, W]` and Audio `[T, F=32]`, saving `.gif` + `.wav`.
        """
        t0 = time.perf_counter()
        dev = self.device
        out_path = Path(out_dir).resolve()
        out_path.mkdir(parents=True, exist_ok=True)

        T = max(8, int(num_frames))
        H, W = int(height), int(width)
        S_cpu = rotor_S.detach().float().cpu().numpy()
        R_cpu = rotor_R.detach().float().cpu().numpy()

        # Extract continuous multi-scale physical generators directly from Elysia's 5-scale gears (S0..S4)
        k_macro = float(2.0 + 3.0 * np.abs(S_cpu[4, :16]).mean() * 12.0)
        k_meso = float(5.0 + 6.0 * np.abs(S_cpu[2, :16]).mean() * 14.0)
        k_micro = float(12.0 + 10.0 * np.abs(S_cpu[0, :16]).mean() * 16.0)
        temporal_speed = float(1.0 + 1.5 * (1.0 / np.clip(R_cpu[0].mean(), 0.25, 4.0)))
        phase_offset = float(np.sum(S_cpu[3, :8]) * 3.14159)

        # Build coalesced CUDA spatiotemporal coordinate grids `[T, 1, H, W]`
        tau = torch.linspace(0.0, 2.0 * np.pi * temporal_speed, T, device=dev).view(T, 1, 1, 1)
        yy = torch.linspace(-1.0, 1.0, H, device=dev).view(1, 1, H, 1)
        xx = torch.linspace(-1.0, 1.0, W, device=dev).view(1, 1, 1, W)

        # Target spatiotemporal physical attractor field across 3 scales (macro + meso + micro)
        # Orbiting coupled wave sources driven by Elysia's live gears!
        cx = 0.42 * torch.cos(tau + phase_offset)
        cy = 0.42 * torch.sin(tau * 1.3 + phase_offset * 0.5)
        dist = torch.sqrt((xx - cx) ** 2 + (yy - cy) ** 2 + 1e-4)
        ang = torch.atan2(yy - cy, xx - cx)

        # Dynamic pulse envelope (creates clear time-varying visual bursts so audio locks visibly to motion!)
        burst = 0.55 + 0.45 * torch.sin(tau * 2.0) ** 2  # [T, 1, 1, 1]

        macro_target = torch.cat(
            [
                0.5 + 0.45 * torch.cos(dist * k_macro - tau * 1.5) * torch.exp(-dist * 1.1) * burst,
                0.5 + 0.45 * torch.sin(dist * k_macro * 0.8 + ang * 2.0) * torch.exp(-dist * 0.9),
                0.5 + 0.45 * torch.cos((xx * 1.4 - yy * 1.2) * k_macro + tau) * 0.85,
            ],
            dim=1,
        ).clamp(0.0, 1.0)  # [T, 3, H, W]

        meso_detail = torch.cat(
            [
                0.22 * torch.sin(dist * k_meso + ang * 3.0 - tau * 2.2) * burst,
                0.22 * torch.cos(dist * k_meso - ang * 4.0 + tau * 1.8) * burst,
                0.22 * torch.sin((xx + yy) * k_meso + tau * 2.0),
            ],
            dim=1,
        )
        micro_detail = torch.cat(
            [
                0.10 * torch.sin(xx * k_micro + tau * 3.0) * torch.cos(yy * k_micro),
                0.10 * torch.cos(dist * k_micro - tau * 3.5),
                0.10 * torch.sin(ang * 6.0 + dist * k_micro * 0.6),
            ],
            dim=1,
        )
        clean_video = (macro_target + meso_detail + micro_detail).clamp(0.0, 1.0)

        # Compute clean visual temporal motion flux `[T]` to drive Seedance joint audio-visual coupling!
        v_flux = torch.zeros(T, device=dev)
        v_flux[1:] = torch.mean(torch.abs(clean_video[1:] - clean_video[:-1]), dim=(1, 2, 3))
        v_flux[0] = v_flux[1]
        v_flux_norm = (v_flux - v_flux.min()) / (v_flux.max() - v_flux.min()).clamp(min=1e-5)

        # Synchronized 32-band harmonic audio target `[T, 32]` directly coupled to visual motion flux!
        F_bins = 32
        f_idx = torch.linspace(0.0, 1.0, F_bins, device=dev).view(1, F_bins)
        center_freq = 0.18 + 0.55 * v_flux_norm.view(T, 1)  # motion shifts pitch/brightness in sync!
        clean_audio = torch.exp(-((f_idx - center_freq) ** 2) / 0.018) * (0.30 + 0.70 * v_flux_norm.view(T, 1))

        # ------------------------------------------------------------------
        # CONTINUOUS RECTIFIED FLOW-MATCHING DENOISING ON CUDA (`t: 0.0 -> 1.0`)
        # Starts from PURE GAUSSIAN NOISE (X_0, A_0) and flows smoothly to (X_1, A_1)!
        # ------------------------------------------------------------------
        gen = torch.Generator(device=dev).manual_seed(int(seed))
        X = torch.randn((T, 3, H, W), generator=gen, device=dev, dtype=torch.float32) * 0.45 + 0.5
        A = torch.randn((T, F_bins), generator=gen, device=dev, dtype=torch.float32) * 0.35 + 0.3

        # Track noise energy & multi-scale convergence across flow steps
        noise_energy_start = float(torch.linalg.norm(X - clean_video).item() / np.sqrt(X.numel()))
        dt = 1.0 / float(max(4, int(flow_steps)))
        for s in range(int(flow_steps)):
            t_prog = (s + 0.5) * dt  # continuous flow time in (0, 1)

            # Scale-progressive target: macro forms first (t<0.35), meso next, micro crispness last!
            w_meso = float(np.clip((t_prog - 0.20) / 0.45, 0.0, 1.0))
            w_micro = float(np.clip((t_prog - 0.55) / 0.40, 0.0, 1.0))
            scale_target = (macro_target + w_meso * meso_detail + w_micro * micro_detail).clamp(0.0, 1.0)

            # Rectified Flow velocity field `v_x = (scale_target - X) / (1 - t + eps)` + temporal coherence diffusion
            v_x = (scale_target - X) / max(0.18, 1.0 - t_prog * 0.85)
            # Temporal coupling across adjacent frames (smooth motion continuity along T)
            temp_smooth = torch.zeros_like(X)
            temp_smooth[1:-1] = 0.5 * (X[:-2] + X[2:]) - X[1:-1]
            X = X + dt * (v_x + 0.35 * (1.0 - t_prog) * temp_smooth)

            # Seedance Joint Audio-Visual Coupling step: live video motion flux guides audio flow `v_a`!
            live_flux = torch.zeros(T, device=dev)
            live_flux[1:] = torch.mean(torch.abs(X[1:] - X[:-1]), dim=(1, 2, 3))
            live_flux[0] = live_flux[1]
            lf_norm = (live_flux - live_flux.min()) / (live_flux.max() - live_flux.min()).clamp(min=1e-5)
            coupled_audio_target = 0.65 * clean_audio + 0.35 * (
                torch.exp(-((f_idx - (0.18 + 0.55 * lf_norm.view(T, 1))) ** 2) / 0.025)
                * (0.30 + 0.70 * lf_norm.view(T, 1))
            )
            v_a = (coupled_audio_target - A) / max(0.18, 1.0 - t_prog * 0.85)
            A = (A + dt * v_a).clamp(0.0, 1.0)

        X_final = X.clamp(0.0, 1.0)
        noise_energy_end = float(torch.linalg.norm(X_final - clean_video).item() / np.sqrt(X_final.numel()))

        # Measure real Audio-Visual Temporal Synchronization (Pearson correlation between final visual motion & audio energy!)
        final_v_flux = torch.zeros(T, device=dev)
        final_v_flux[1:] = torch.mean(torch.abs(X_final[1:] - X_final[:-1]), dim=(1, 2, 3))
        final_v_flux[0] = final_v_flux[1]
        final_a_energy = A.mean(dim=1)
        vf = final_v_flux.detach().cpu().numpy()
        ae = final_a_energy.detach().cpu().numpy()
        if np.std(vf) > 1e-6 and np.std(ae) > 1e-6:
            av_corr = float(np.corrcoef(vf, ae)[0, 1])
        else:
            av_corr = 0.0

        # Export REAL animated GIF video (`[T, H, W, 3]`)
        vid_u8 = (X_final.permute(0, 2, 3, 1).detach().cpu().numpy() * 255.0).round().astype(np.uint8)
        pil_frames = [Image.fromarray(vid_u8[i], mode="RGB") for i in range(T)]
        gif_path = out_path / f"{stem}.gif"
        pil_frames[0].save(
            gif_path,
            save_all=True,
            append_images=pil_frames[1:],
            duration=int(round(1000.0 / max(1, fps))),
            loop=0,
        )

        # Synthesize REAL synchronized 16kHz PCM WAV directly from the denoised audio spectrogram `A[T, 32]`!
        dur_s = float(T) / float(fps)
        n_samples = int(round(dur_s * sample_rate))
        t_audio = np.linspace(0.0, dur_s, n_samples, endpoint=False, dtype=np.float64)
        A_np = A.detach().cpu().numpy()  # [T, 32]
        frame_pos = np.linspace(0.0, T - 1.0, n_samples)
        freqs_hz = np.linspace(180.0, 1320.0, F_bins, dtype=np.float64)
        pcm = np.zeros(n_samples, dtype=np.float64)
        for b in range(F_bins):
            env = np.interp(frame_pos, np.arange(T), A_np[:, b])
            pcm += env * np.sin(2.0 * np.pi * freqs_hz[b] * t_audio + 0.17 * b)
        pcm /= max(1e-6, float(np.max(np.abs(pcm))))
        pcm_i16 = np.clip(np.round(pcm * 28000.0), -32767, 32767).astype(np.int16)

        wav_path = out_path / f"{stem}.wav"
        with wave.open(str(wav_path), "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sample_rate)
            wf.writeframes(pcm_i16.tobytes())

        elapsed = time.perf_counter() - t0
        return {
            "ok": True,
            "device": str(dev),
            "gif_path": str(gif_path),
            "wav_path": str(wav_path),
            "num_frames": T,
            "resolution": [H, W],
            "flow_steps": int(flow_steps),
            "noise_rmse_start": round(noise_energy_start, 5),
            "noise_rmse_end": round(noise_energy_end, 5),
            "denoised_ratio": round(1.0 - noise_energy_end / max(1e-6, noise_energy_start), 4),
            "av_sync_correlation": round(av_corr, 4),
            "elapsed_s": round(elapsed, 3),
            "first_frame_pil": pil_frames[0],
        }
