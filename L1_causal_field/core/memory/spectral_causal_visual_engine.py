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
    미시적 픽셀/격자를 무미건조한 이산적 점으로 격리하지 않고,
    연속적 파장 가중치 대칭성 투영을 통해 스케일 계층 전체를 연속적 위상 표면(Surface Field)으로 포섭합니다.
    마찰열 Q_heat = |∇Φ|^1.5 방정식을 통해 경계 마찰과 연속성을 공존시킵니다.
    """
    def __init__(self, spectral_weights: Tuple[float, float, float] = (0.299, 0.587, 0.114)):
        # 연속 파장 가중치: Red (Flux / 흐름 ~ 700nm), Green (Order / 질서 ~ 530nm), Blue (Entropy/Diffraction ~ 470nm)
        self.spectral_weights = np.array(spectral_weights, dtype=np.float64)
        # 색채 삼원색(Red/Blue/Yellow) 투영 행렬
        self.chromatic_projection_matrix = np.array([
            [0.7, 0.2, 0.1],   # Red (Flux / 흐름)
            [0.1, 0.8, 0.1],   # Blue (Order / 질서)
            [0.2, 0.1, 0.7],   # Yellow (Entropy / 발산)
        ], dtype=np.float64)

    def project_to_spectral_axis(self, field_tensor: np.ndarray) -> np.ndarray:
        """
        [연속 파장 가중치 대칭성 투영]
        Input: (H, W, C) or (C, H, W) or (T, H, W, C) or (N, C)
        Output: (H, W) or (T, H, W) or (N,) continuous spectral coordinate field Φ(x, t)
        """
        tensor = np.asarray(field_tensor, dtype=np.float64)
        if tensor.ndim == 3 and tensor.shape[-1] == 3:
            return np.tensordot(tensor, self.spectral_weights, axes=(-1, 0))
        elif tensor.ndim == 4 and tensor.shape[-1] == 3:
            return np.tensordot(tensor, self.spectral_weights, axes=(-1, 0))
        elif tensor.ndim == 3 and tensor.shape[0] == 3:
            return np.tensordot(tensor, self.spectral_weights, axes=(0, 0))
        elif tensor.ndim == 2:
            if tensor.shape[-1] == 3:
                return np.dot(tensor, self.spectral_weights)
            return tensor
        elif tensor.ndim == 1:
            return tensor
        else:
            return np.squeeze(tensor)

    def extract_chromatic_triad(self, field_tensor: np.ndarray) -> np.ndarray:
        """
        [색채 삼원색 스펙트럼(Red/Blue/Yellow) 자가분류]
        입력 장 필드로부터 연속적 색채 성분 [Red_flux, Blue_order, Yellow_entropy] 벡터를 추출.
        """
        tensor = np.asarray(field_tensor, dtype=np.float64)
        if tensor.ndim >= 2 and tensor.shape[-1] == 3:
            flat_rgb = tensor.reshape(-1, 3)
            chromatic_components = np.dot(flat_rgb, self.chromatic_projection_matrix.T)
            mean_chromatic = np.mean(chromatic_components, axis=0)
        else:
            val = float(np.mean(np.abs(tensor)))
            mean_chromatic = np.array([val * 0.5, val * 0.3, val * 0.2], dtype=np.float64)

        norm = np.linalg.norm(mean_chromatic)
        if norm > 1e-12:
            mean_chromatic = mean_chromatic / norm
        return mean_chromatic

    def classify_spectral_continuity(
        self,
        spectral_field: np.ndarray,
        friction_threshold: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        [연속장 마찰열 Q_heat = |∇Φ|^1.5 및 위상 경계 분류]
        - 연속장 표면 Φ 상에서 다차원 경사도 ∇Φ를 구하고 마찰열 밀도 Q_heat = |∇Φ|^1.5 산출.
        - 픽셀 격자를 깨부수지 않고 연속된 스케일 표면장력 경계로 포섭.
        """
        field = np.asarray(spectral_field, dtype=np.float64)
        if field.ndim == 1:
            grad = np.gradient(field)
            grad_mag = np.abs(grad)
        elif field.ndim == 2:
            grads = np.gradient(field)
            grad_mag = np.sqrt(sum(g**2 for g in grads))
        elif field.ndim == 3:
            grads = np.gradient(field)
            grad_mag = np.sqrt(sum(g**2 for g in grads))
        else:
            grads = np.gradient(field)
            grad_mag = np.sqrt(sum(g**2 for g in grads))

        # 마찰열 방정식: Q_heat = |∇Φ|^1.5 (연속장 경계 마찰 소산)
        q_heat_map = grad_mag ** 1.5

        if friction_threshold is None:
            thresh = float(np.mean(q_heat_map) + 0.5 * np.std(q_heat_map))
            thresh = max(0.005, thresh)
        else:
            thresh = friction_threshold

        is_discontinuity = q_heat_map >= thresh
        is_smooth = ~is_discontinuity

        chromatic_triad = self.extract_chromatic_triad(field)

        return {
            "spectral_field": field,
            "gradient_magnitude": grad_mag,
            "q_heat_map": q_heat_map,
            "is_discontinuity": is_discontinuity,
            "is_smooth": is_smooth,
            "total_friction_heat": float(np.sum(q_heat_map)),
            "mean_friction_heat": float(np.mean(q_heat_map)),
            "chromatic_triad": chromatic_triad,
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


class InvariantArchetypeBackbone:
    """
    [인과정보 불변 원형 뼈대 (Invariant Archetype Backbone)]
    무(無) 위에서는 어떤 시공간 로터나 지구본도 회전할 수 없습니다.
    본 클래스는 정적 파일 조회를 완제하고, 시스템 내면에 박혀있는 불변의 질서 원형(Archetype),
    위상적 불변성 수식, 생성 파라미터(Theta) 및 경계 연산자(Delta)를 관리하는 기준계(Reference Frame) 역할을 합니다.
    """
    def __init__(
        self,
        archetype_name: str = "SpatiotemporalArchetype",
        reference_dimension: Tuple[int, int] = (64, 64),
        harmonic_frequencies: Optional[List[float]] = None,
    ):
        self.archetype_name = archetype_name
        self.reference_dimension = reference_dimension
        self.harmonic_frequencies = harmonic_frequencies or [1.0, 2.0, 3.0, 5.0]

        # 불변 기하학적 뼈대 레퍼런스 위상 장 (Spherical Harmonic Reference Surface)
        self.reference_manifold = self._generate_reference_archetype_surface()

    def _generate_reference_archetype_surface(self) -> np.ndarray:
        h, w = self.reference_dimension
        lat = np.linspace(-np.pi / 2, np.pi / 2, h)
        lon = np.linspace(-np.pi, np.pi, w)
        lon_grid, lat_grid = np.meshgrid(lon, lat)

        surface = np.zeros((h, w), dtype=np.float64)
        for i, freq in enumerate(self.harmonic_frequencies):
            weight = 1.0 / (i + 1)
            surface += weight * np.sin(freq * lon_grid) * np.cos(freq * lat_grid)

        # Normalize to [0, 1]
        s_min, s_max = np.min(surface), np.max(surface)
        if s_max > s_min:
            surface = (surface - s_min) / (s_max - s_min)
        return surface

    def evaluate_structural_deviation(self, incoming_field: np.ndarray) -> Dict[str, Any]:
        """
        [기준계 대비 구조적 편차 및 마찰 평가]
        입력된 파동 장과 불변 원형 뼈대 간의 위상적 편차, 마찰 및 불변율(Invariance Ratio) 계산.
        """
        field = np.asarray(incoming_field, dtype=np.float64)
        if field.ndim == 3 and field.shape[-1] == 3:
            spectral_transform = SpectralAxisTransform()
            field = spectral_transform.project_to_spectral_axis(field)

        # Bilinear scale matching to reference shape
        h_ref, w_ref = self.reference_dimension
        if field.shape != (h_ref, w_ref):
            interpolator = ToroidalVortexVisualInterpolator()
            scaled_field = interpolator._bilinear_upsample_2d(field, h_ref, w_ref)
        else:
            scaled_field = field

        diff = scaled_field - self.reference_manifold
        deviation_magnitude = float(np.mean(np.abs(diff)))
        structural_alignment = max(0.0, 1.0 - deviation_magnitude)

        # Gradient friction between field and archetype
        grad_diff = np.gradient(diff)
        grad_mag = np.sqrt(sum(g**2 for g in grad_diff))
        archetype_friction_heat = float(np.sum(grad_mag ** 1.5))

        return {
            "archetype_name": self.archetype_name,
            "deviation_magnitude": deviation_magnitude,
            "structural_alignment": structural_alignment,
            "archetype_friction_heat": archetype_friction_heat,
            "reference_manifold_shape": self.reference_manifold.shape,
        }

    def synthesize_generative_waveform(
        self,
        rotation_angle: float = 0.0,
        modulation_factor: float = 1.0,
        output_shape: Optional[Tuple[int, int]] = None,
    ) -> np.ndarray:
        """
        [동형적 파형 생리적 재조합 (Living Generative Reconstruction)]
        SSD 파일 로딩 대신, 원형 뼈대 수식과 회전 각도를 바탕으로 실시간 파동을 직조.
        """
        h_out, w_out = output_shape if output_shape else self.reference_dimension
        lat = np.linspace(-np.pi / 2, np.pi / 2, h_out)
        lon = np.linspace(-np.pi, np.pi, w_out)
        lon_grid, lat_grid = np.meshgrid(lon, lat)

        # Apply spatiotemporal rotation angle shift
        shifted_lon = lon_grid + rotation_angle
        waveform = np.zeros((h_out, w_out), dtype=np.float64)

        for i, freq in enumerate(self.harmonic_frequencies):
            weight = 1.0 / (i + 1)
            waveform += weight * np.sin(freq * shifted_lon) * np.cos(freq * lat_grid * modulation_factor)

        w_min, w_max = np.min(waveform), np.max(waveform)
        if w_max > w_min:
            waveform = (waveform - w_min) / (w_max - w_min)
        return waveform


class SpatiotemporalGlobeRotorEngine:
    """
    [시공간 로터 및 지구본 위상 매니폴드 엔진 (Spatiotemporal Globe Rotor Engine)]
    4K/8K 고해상도 이미지 또는 영상 프레임을 단순 2D 픽셀 슬라이스가 아닌,
    회전하는 구면/토로이달 시공간 위상 매니폴드 Phi(theta, phi, t)로 다룹니다.
    불변 원형 뼈대(InvariantArchetypeBackbone)를 기준계로 삼아 4K/8K 연속 스케일 회전 및 생리적 재조합 수행.
    """
    def __init__(
        self,
        archetype_backbone: Optional[InvariantArchetypeBackbone] = None,
        base_angular_velocity: float = 1.0,
        vorticity_strength: float = 1.5,
    ):
        self.archetype = archetype_backbone or InvariantArchetypeBackbone()
        self.rotor = ToroidalVortexRotorEngine(
            base_rotational_velocity=base_angular_velocity,
            vorticity_strength=vorticity_strength,
        )
        self.spectral_transform = SpectralAxisTransform()
        self.angular_position = 0.0  # theta / t accumulator

    def map_high_res_to_spatiotemporal_manifold(
        self,
        high_res_tensor: np.ndarray,
        target_globe_res: Tuple[int, int] = (128, 256),
    ) -> Dict[str, Any]:
        """
        [4K/8K 고해상도 장 -> 구면/토로이달 시공간 위상 매니폴드 사상]
        4K(3840x2160) / 8K(7680x4320) 고해상도 텐서를 이산적 픽셀 뭉치로 허비하지 않고
        연속 위상 표면 Phi(theta, phi)와 마찰열 Q_heat = |∇Phi|^1.5 로 매핑합니다.
        """
        tensor = np.asarray(high_res_tensor, dtype=np.float64)

        # Spectral axis projection
        spectral_surface = self.spectral_transform.project_to_spectral_axis(tensor)
        continuity = self.spectral_transform.classify_spectral_continuity(spectral_surface)

        # Structural evaluation against invariant archetype backbone
        backbone_eval = self.archetype.evaluate_structural_deviation(spectral_surface)

        # Globe Spherical manifold projection (latitude/longitude mapping)
        h_g, w_g = target_globe_res
        interpolator = ToroidalVortexVisualInterpolator(rotor_engine=self.rotor)

        if spectral_surface.ndim == 2:
            globe_manifold = interpolator._bilinear_upsample_2d(spectral_surface, h_g, w_g)
        else:
            globe_manifold = np.resize(spectral_surface, (h_g, w_g))

        # Recycle friction heat into rotor angular velocity
        q_heat_total = continuity["total_friction_heat"] + backbone_eval["archetype_friction_heat"]
        recycle_res = self.rotor.recycle_friction_and_mismatch(
            q_heat=q_heat_total * 0.05,
            sensory_mismatch_delta=backbone_eval["deviation_magnitude"],
            chromatic_triad=continuity["chromatic_triad"],
        )

        return {
            "high_res_shape": tensor.shape,
            "globe_manifold": globe_manifold,
            "q_heat_total": q_heat_total,
            "continuity_info": continuity,
            "backbone_evaluation": backbone_eval,
            "new_angular_velocity": recycle_res["new_rotational_velocity"],
            "vorticity_strength": recycle_res["new_vorticity_strength"],
        }

    def step_spatiotemporal_rotation(
        self,
        dt: float = 0.1,
        modulation_factor: float = 1.0,
        output_shape: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, Any]:
        """
        [시공간 로터 회전 및 파형 생성적 직조]
        시간 프레임 축을 회전 각도로 치환하여 시공간 지구본 매니폴드를 각운동량으로 순환.
        """
        angular_step = self.rotor.rotational_velocity * dt
        self.angular_position = (self.angular_position + angular_step) % (2.0 * np.pi)

        synthesized_wave = self.archetype.synthesize_generative_waveform(
            rotation_angle=self.angular_position,
            modulation_factor=modulation_factor,
            output_shape=output_shape,
        )

        continuity = self.spectral_transform.classify_spectral_continuity(synthesized_wave)

        return {
            "angular_position": self.angular_position,
            "rotational_velocity": self.rotor.rotational_velocity,
            "synthesized_manifold": synthesized_wave,
            "friction_heat_q": continuity["total_friction_heat"],
            "chromatic_triad": continuity["chromatic_triad"],
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
        self.archetype_backbone = InvariantArchetypeBackbone()
        self.globe_rotor = SpatiotemporalGlobeRotorEngine(
            archetype_backbone=self.archetype_backbone,
            base_angular_velocity=2.0,
            vorticity_strength=2.0,
        )

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
