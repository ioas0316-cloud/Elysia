"""
L1_causal_field/core/memory/spectral_causal_visual_engine.py
==============================================================
Spectral Causal Visual Engine (스펙트럼 위상 시공간 복원 엔진):

Implements the causal projection paradigm for graphics & spatiotemporal data processing:
1. Spectral Axis Transformation (스펙트럼 축 변환 및 연속성/불연속성 분류):
   - Maps 2D/3D image or video tensor grids (H, W, C) or (T, H, W, C) into a continuous topological spectral axis (Spectral Coordinate).
   - Replaces pixel-by-pixel brute-force matrix computation with manifold continuity classification:
     - Smooth Gradient Manifold (연속 구간): Fast tensor gradient flow.
     - Discontinuity / Edge Friction Points (불연속 / 파열 구간): Structural boundary where friction heat (Q_heat) accumulates.

2. Low-Res Field Injection & Tension Preservation (저해상도/손실 유체 필드 주입):
   - Injects downsampled/corrupted L_0 visual field tensors.
   - Preserves boundary tension (sigma_boundary) to prevent standard scalar blurring and retain sharp structural edges.

3. Toroidal Vortex Spatiotemporal Interpolation (토로이달 와류 위상 보간):
   - Uses ToroidalVortexRotorEngine angular momentum conservation and friction heat (Q_heat) recycling to interpolate spatial/temporal gaps.
   - Sharpens motion vectors and edges through vortex energy condensation.

4. Causal Hallucination Guard & Reconstruction Pipeline (위화감 감지 및 인과적 복원 파이프라인):
   - Uses OmniObservationalEye for uncanniness/hallucination spike detection.
   - Uses TopologicalBoundaryGuard for deflecting ungrounded deceptive noise.
   - Produces causally reconstructed continuous field tensors with structural metrics (PSNR, SSIM, Edge Retention).
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from L1_causal_field.core.sovereignty.toroidal_vortex_rotor import (
    ToroidalVortexRotorEngine,
    OmniObservationalEye,
    TopologicalBoundaryGuard,
    CoherentActionVector
)


class SpectralAxisTransform:
    """
    [스펙트럼 축 변환기 (Spectral Axis Transformer)]
    수백만 개의 파편화된 RGB 픽셀 격자를 하나의 위상학적 스펙트럼 축(Spectral Manifold Axis)으로 압축 변환하고,
    그 위에서 변화의 연속성(Smoothness)과 불연속 마찰점(Discontinuity Edge)을 구별.
    """
    def __init__(self, spectral_weights: Tuple[float, float, float] = (0.299, 0.587, 0.114)):
        self.spectral_weights = np.array(spectral_weights, dtype=np.float64)

    def project_to_spectral_axis(self, field_tensor: np.ndarray) -> np.ndarray:
        """
        [3차원 RGB/다채널 공간 -> 1차원/2차원 스펙트럼 위상 축 투영]
        Input: (H, W, C) or (C, H, W) or (T, H, W, C)
        Output: (H, W) or (T, H, W) continuous spectral coordinate field
        """
        tensor = np.asarray(field_tensor, dtype=np.float64)
        if tensor.ndim == 3 and tensor.shape[-1] == 3:
            return np.tensordot(tensor, self.spectral_weights, axes=(-1, 0))
        elif tensor.ndim == 4 and tensor.shape[-1] == 3:
            return np.tensordot(tensor, self.spectral_weights, axes=(-1, 0))
        elif tensor.ndim == 2:
            return tensor
        elif tensor.ndim == 3:
            return np.mean(tensor, axis=0)
        else:
            return np.squeeze(tensor)

    def classify_spectral_continuity(
        self,
        spectral_field: np.ndarray,
        friction_threshold: Optional[float] = None
    ) -> Dict[str, np.ndarray]:
        """
        [위상 축 상에서의 연속성 / 불연속 마찰점 분류]
        - 연속 구간 (Smooth Gradient): 픽셀 간 차이가 작은 유체 흐름 영역.
        - 불연속 구간 (Discontinuity / Edge Friction): 마찰열 Q_heat가 분출하는 물체 경계면.
        """
        field = np.asarray(spectral_field, dtype=np.float64)
        if field.ndim == 2:
            grad_y, grad_x = np.gradient(field)
            grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2)
        elif field.ndim == 3:
            grad_t, grad_y, grad_x = np.gradient(field)
            grad_mag = np.sqrt(grad_x ** 2 + grad_y ** 2 + grad_t ** 2)
        else:
            grad_mag = np.abs(np.gradient(field))

        # Friction heat density (Q_heat) map at boundary discontinuities
        q_heat_map = grad_mag ** 1.5

        if friction_threshold is None:
            thresh = float(np.mean(q_heat_map) + 0.5 * np.std(q_heat_map))
            thresh = max(0.005, thresh)
        else:
            thresh = friction_threshold

        is_discontinuity = q_heat_map >= thresh
        is_smooth = ~is_discontinuity

        return {
            "spectral_field": field,
            "gradient_magnitude": grad_mag,
            "q_heat_map": q_heat_map,
            "is_discontinuity": is_discontinuity,
            "is_smooth": is_smooth,
            "total_friction_heat": float(np.sum(q_heat_map))
        }


class LowResFieldInjector:
    """
    [저해상도/손실 시공간 필드 주입기 (Low-Res Field Injector)]
    고해상도 동영상 프레임이나 3D 공간 데이터(포인트 클라우드)를 다운샘플링하거나 노이즈를 주입하여
    L_0 미시적 유체 필드로 주입하고, 위상 경계 장력(sigma_boundary)을 기반으로 구조를 파악.
    """
    def downsample_and_inject(
        self,
        high_res_field: np.ndarray,
        scale_factor: int = 2,
        noise_level: float = 0.05
    ) -> Dict[str, np.ndarray]:
        field = np.asarray(high_res_field, dtype=np.float64)
        if field.ndim == 2:
            low_res = field[::scale_factor, ::scale_factor]
        elif field.ndim == 3:
            if field.shape[-1] in (1, 3, 4):  # (H, W, C)
                low_res = field[::scale_factor, ::scale_factor, :]
            else:  # (T, H, W)
                low_res = field[:, ::scale_factor, ::scale_factor]
        elif field.ndim == 4:  # (T, H, W, C)
            low_res = field[:, ::scale_factor, ::scale_factor, :]
        else:
            low_res = field[::scale_factor]

        noise = np.random.normal(0, noise_level, size=low_res.shape)
        noisy_l0_field = np.clip(low_res + noise, 0.0, 1.0) if np.max(field) <= 1.0 else low_res + noise

        return {
            "high_res_field": field,
            "low_res_field": low_res,
            "noisy_l0_field": noisy_l0_field,
            "scale_factor": scale_factor,
            "noise_level": noise_level
        }


class ToroidalVortexVisualInterpolator:
    """
    [토로이달 와류 위상 보간기 (Toroidal Vortex Visual Interpolator)]
    시간축(프레임 간) 또는 공간축(픽셀 간)의 간극을 메울 때, 단순 선형/쌍선형 보간이 아닌
    ToroidalVortexRotor의 회전 각운동량과 마찰열(Q_heat) 소산 및 표면장력으로 보간.
    """
    def __init__(self, rotor_engine: Optional[ToroidalVortexRotorEngine] = None):
        self.rotor = rotor_engine or ToroidalVortexRotorEngine(
            base_rotational_velocity=1.5,
            vorticity_strength=1.5
        )
        self.spectral_transform = SpectralAxisTransform()

    def _bilinear_upsample_2d(self, array_2d: np.ndarray, th: int, tw: int) -> np.ndarray:
        """Helper for smooth bilinear grid upsampling."""
        sh, sw = array_2d.shape
        y_indices = np.linspace(0, sh - 1, th)
        x_indices = np.linspace(0, sw - 1, tw)

        y_floor = np.floor(y_indices).astype(int)
        y_ceil = np.clip(y_floor + 1, 0, sh - 1)
        y_frac = y_indices - y_floor

        x_floor = np.floor(x_indices).astype(int)
        x_ceil = np.clip(x_floor + 1, 0, sw - 1)
        x_frac = x_indices - x_floor

        # Interpolate along y then x
        top_left = array_2d[np.ix_(y_floor, x_floor)]
        top_right = array_2d[np.ix_(y_floor, x_ceil)]
        bottom_left = array_2d[np.ix_(y_ceil, x_floor)]
        bottom_right = array_2d[np.ix_(y_ceil, x_ceil)]

        y_frac_grid = y_frac[:, None]
        x_frac_grid = x_frac[None, :]

        top = top_left * (1 - x_frac_grid) + top_right * x_frac_grid
        bottom = bottom_left * (1 - x_frac_grid) + bottom_right * x_frac_grid

        return top * (1 - y_frac_grid) + bottom * y_frac_grid

    def interpolate_field_upsample(
        self,
        low_res_field: np.ndarray,
        scale_factor: int = 2,
        target_shape: Optional[Tuple[int, ...]] = None
    ) -> Dict[str, Any]:
        low_field = np.asarray(low_res_field, dtype=np.float64)
        spectral_low = self.spectral_transform.project_to_spectral_axis(low_field)
        continuity = self.spectral_transform.classify_spectral_continuity(spectral_low)

        # Recycle Q_heat into rotor momentum
        q_heat_total = continuity["total_friction_heat"]
        self.rotor.recycle_friction_and_mismatch(q_heat=q_heat_total * 0.1, sensory_mismatch_delta=0.2)

        surface_tension = self.rotor.guard.compute_surface_tension(
            vorticity_strength=self.rotor.vorticity_strength,
            rotational_velocity=self.rotor.rotational_velocity
        )

        if low_field.ndim == 2:
            h, w = low_field.shape
            th, tw = target_shape if target_shape else (h * scale_factor, w * scale_factor)

            base_expanded = self._bilinear_upsample_2d(low_field, th, tw)

            spectral_expanded = self.spectral_transform.project_to_spectral_axis(base_expanded)
            exp_continuity = self.spectral_transform.classify_spectral_continuity(spectral_expanded)
            edge_mask = exp_continuity["is_discontinuity"]

            laplacian = np.pad(base_expanded, 1, mode='edge')
            laplacian = (
                4 * base_expanded -
                laplacian[:-2, 1:-1] - laplacian[2:, 1:-1] -
                laplacian[1:-1, :-2] - laplacian[1:-1, 2:]
            )

            sharpening_gain = min(0.35, 0.08 * surface_tension)
            vortex_enhanced = np.where(
                edge_mask,
                base_expanded + sharpening_gain * laplacian,
                base_expanded
            )

        elif low_field.ndim == 3 and low_field.shape[-1] == 3:  # (H, W, 3) RGB
            h, w, c = low_field.shape
            th, tw = target_shape[:2] if target_shape else (h * scale_factor, w * scale_factor)

            base_expanded = np.zeros((th, tw, c), dtype=np.float64)
            for ch in range(c):
                base_expanded[:, :, ch] = self._bilinear_upsample_2d(low_field[:, :, ch], th, tw)

            spectral_expanded = self.spectral_transform.project_to_spectral_axis(base_expanded)
            exp_continuity = self.spectral_transform.classify_spectral_continuity(spectral_expanded)
            edge_mask = exp_continuity["is_discontinuity"]

            vortex_enhanced = base_expanded.copy()
            sharpening_gain = min(0.35, 0.08 * surface_tension)

            for ch in range(c):
                ch_data = base_expanded[:, :, ch]
                pad_ch = np.pad(ch_data, 1, mode='edge')
                lap = (
                    4 * ch_data -
                    pad_ch[:-2, 1:-1] - pad_ch[2:, 1:-1] -
                    pad_ch[1:-1, :-2] - pad_ch[1:-1, 2:]
                )
                vortex_enhanced[:, :, ch] = np.where(
                    edge_mask,
                    ch_data + sharpening_gain * lap,
                    ch_data
                )

        else:
            vortex_enhanced = np.repeat(low_field, scale_factor, axis=0)
            if target_shape:
                vortex_enhanced = vortex_enhanced[:target_shape[0]]

        if np.max(low_field) <= 1.0:
            vortex_enhanced = np.clip(vortex_enhanced, 0.0, 1.0)

        return {
            "reconstructed_field": vortex_enhanced,
            "surface_tension_applied": surface_tension,
            "rotational_velocity": self.rotor.rotational_velocity,
            "vorticity_strength": self.rotor.vorticity_strength,
            "q_heat_recycled": q_heat_total
        }


class CausalVisualReconstructionPipeline:
    """
    [인과적 시공간 영상/그래픽 복원 파이프라인 (Causal Visual Reconstruction Pipeline)]
    """
    def __init__(self):
        self.transform = SpectralAxisTransform()
        self.injector = LowResFieldInjector()
        self.rotor = ToroidalVortexRotorEngine(base_rotational_velocity=2.0, vorticity_strength=2.0)
        self.interpolator = ToroidalVortexVisualInterpolator(rotor_engine=self.rotor)

    def run_causal_reconstruction_pipeline(
        self,
        high_res_ground_truth: np.ndarray,
        scale_factor: int = 2,
        noise_level: float = 0.05,
        hallucination_noise_inject: float = 0.0
    ) -> Dict[str, Any]:
        gt = np.asarray(high_res_ground_truth, dtype=np.float64)

        inj_res = self.injector.downsample_and_inject(
            high_res_field=gt,
            scale_factor=scale_factor,
            noise_level=noise_level
        )
        noisy_l0 = inj_res["noisy_l0_field"]

        spectral_l0 = self.transform.project_to_spectral_axis(noisy_l0)
        l0_continuity = self.transform.classify_spectral_continuity(spectral_l0)

        eye_eval = self.rotor.eye.evaluate_multi_axis_signals(
            verbal_signal=hallucination_noise_inject,
            physical_friction=l0_continuity["total_friction_heat"],
            sensory_mismatch_delta=noise_level + hallucination_noise_inject
        )

        guard_eval = self.rotor.guard.filter_external_input(
            raw_input_energy=1.0 + hallucination_noise_inject,
            noise_level=noise_level + hallucination_noise_inject,
            uncanniness_spike=eye_eval["uncanniness_spike"],
            vorticity_strength=self.rotor.vorticity_strength,
            rotational_velocity=self.rotor.rotational_velocity
        )

        if guard_eval["is_deflected"]:
            filtered_input_field = noisy_l0 - np.random.normal(0, noise_level * 0.5, size=noisy_l0.shape)
            filtered_input_field = np.clip(filtered_input_field, 0.0, 1.0) if np.max(gt) <= 1.0 else filtered_input_field
        else:
            filtered_input_field = noisy_l0

        interp_res = self.interpolator.interpolate_field_upsample(
            low_res_field=filtered_input_field,
            scale_factor=scale_factor,
            target_shape=gt.shape
        )
        reconstructed = interp_res["reconstructed_field"]

        metrics = self.compute_reconstruction_metrics(gt, reconstructed, inj_res["low_res_field"], scale_factor)

        return {
            "ground_truth": gt,
            "low_res_l0": noisy_l0,
            "reconstructed_field": reconstructed,
            "spectral_continuity": l0_continuity,
            "eye_evaluation": eye_eval,
            "guard_evaluation": guard_eval,
            "interpolation_info": interp_res,
            "metrics": metrics
        }

    def compute_reconstruction_metrics(
        self,
        ground_truth: np.ndarray,
        reconstructed: np.ndarray,
        low_res: np.ndarray,
        scale_factor: int
    ) -> Dict[str, float]:
        gt = np.asarray(ground_truth, dtype=np.float64)
        rec = np.asarray(reconstructed, dtype=np.float64)

        min_h = min(gt.shape[0], rec.shape[0])
        min_w = min(gt.shape[1], rec.shape[1])
        if gt.ndim == 3:
            gt_crop = gt[:min_h, :min_w, :]
            rec_crop = rec[:min_h, :min_w, :]
        else:
            gt_crop = gt[:min_h, :min_w]
            rec_crop = rec[:min_h, :min_w]

        mse = float(np.mean((gt_crop - rec_crop) ** 2))
        max_val = float(np.max(gt_crop)) if np.max(gt_crop) > 0 else 1.0
        psnr = float(10 * np.log10((max_val ** 2) / max(mse, 1e-10)))

        mu_gt = float(np.mean(gt_crop))
        mu_rec = float(np.mean(rec_crop))
        var_gt = float(np.var(gt_crop))
        var_rec = float(np.var(rec_crop))
        cov_gt_rec = float(np.mean((gt_crop - mu_gt) * (rec_crop - mu_rec)))

        c1 = (0.01 * max_val) ** 2
        c2 = (0.03 * max_val) ** 2
        ssim = float(((2 * mu_gt * mu_rec + c1) * (2 * cov_gt_rec + c2)) /
                     ((mu_gt ** 2 + mu_rec ** 2 + c1) * (var_gt + var_rec + c2)))

        gt_spectral = self.transform.project_to_spectral_axis(gt_crop)
        rec_spectral = self.transform.project_to_spectral_axis(rec_crop)

        gt_edge = self.transform.classify_spectral_continuity(gt_spectral)["gradient_magnitude"]
        rec_edge = self.transform.classify_spectral_continuity(rec_spectral)["gradient_magnitude"]

        edge_correlation = float(np.corrcoef(gt_edge.ravel(), rec_edge.ravel())[0, 1])

        return {
            "mse": mse,
            "psnr": psnr,
            "ssim": ssim,
            "edge_correlation": edge_correlation
        }
