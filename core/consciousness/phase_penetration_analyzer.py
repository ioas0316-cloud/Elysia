"""
Phase Penetration Analyzer & 3D Tensor Field Visualizers
========================================================
경계면(d_Omega)에서 유입된 위상 플럭스 J_flux가 내부 매니폴드로 스묘드는 특성 거리(delta_phase)
및 감쇄 계수(alpha)를 계산하고, 3D 스칼라/벡터장 및 시공간 감쇄 히트맵을 시각화하는 모듈입니다.
"""

import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.ndimage import distance_transform_edt
from typing import Dict, Any, Tuple, Optional, List

try:
    from skimage.measure import marching_cubes
except ImportError:
    marching_cubes = None

try:
    import pyvista as pv
except ImportError:
    pv = None


class PhasePenetrationAnalyzer:
    """
    3D 격자 내 외부 경계 플럭스(J_flux)의 위상 침투 깊이(delta_phase)를 측정하는 분석기
    """

    def __init__(self, shape: Tuple[int, int, int], dx: float = 0.1):
        self.depth, self.height, self.width = shape
        self.dx = dx

        # 1. 경계면 d_Omega 정의 (경계=False, 내부=True)
        interior_mask = np.ones(shape, dtype=bool)
        interior_mask[0, :, :] = False
        interior_mask[-1, :, :] = False
        interior_mask[:, 0, :] = False
        interior_mask[:, -1, :] = False
        interior_mask[:, :, 0] = False
        interior_mask[:, :, -1] = False

        # 2. 유클리드 거리 변환 (EDT)
        distance_map_grid = distance_transform_edt(interior_mask)
        self.distance_map = distance_map_grid * dx

        self.max_depth_index = int(np.max(distance_map_grid))
        self.depth_indices = [np.where(distance_map_grid == i) for i in range(self.max_depth_index + 1)]
        self.distances = np.arange(self.max_depth_index + 1) * self.dx

    @torch.no_grad()
    def analyze(self, j_flux_tensor: torch.Tensor) -> Dict[str, Any]:
        """
        Args:
            j_flux_tensor: [D, H, W, 4] 또는 [D, H, W] (유입 플럭스 텐서)
        Returns:
            dict: {
                "delta_phase": 1/e 특성 침투 깊이,
                "delta_phase_fit": 로그-선형 피팅 침투 깊이,
                "attenuation_coeff": 위상 감쇄 계수 (alpha),
                "j_profile": 깊이 d별 평균 플럭스 밀도,
                "distances": 깊이 축 실거리 배열
            }
        """
        if hasattr(j_flux_tensor, "cpu"):
            j_flux = j_flux_tensor.cpu().numpy()
        else:
            j_flux = np.array(j_flux_tensor)

        if j_flux.ndim == 4 and j_flux.shape[-1] == 4:
            j_flux_norm = j_flux.copy()
            j_flux_norm[..., 0] -= 1.0
            flux_magnitude = np.linalg.norm(j_flux_norm, axis=-1)
        else:
            flux_magnitude = np.abs(j_flux)

        j_profile = np.zeros(self.max_depth_index + 1)
        for d_idx, coords in enumerate(self.depth_indices):
            if len(coords[0]) > 0:
                j_profile[d_idx] = np.mean(flux_magnitude[coords])

        j_0 = j_profile[0]
        threshold_1_e = j_0 * (1.0 / math.e)

        below_thresh_indices = np.where(j_profile <= threshold_1_e)[0]
        if len(below_thresh_indices) > 0 and below_thresh_indices[0] > 0:
            idx = below_thresh_indices[0]
            d1, d2 = self.distances[idx - 1], self.distances[idx]
            j1, j2 = j_profile[idx - 1], j_profile[idx]

            if j1 != j2:
                delta_phase = float(d1 + (threshold_1_e - j1) * (d2 - d1) / (j2 - j1))
            else:
                delta_phase = float(d1)
        else:
            delta_phase = float(self.distances[-1])

        valid_mask = j_profile > 1e-8
        if np.sum(valid_mask) >= 2:
            d_valid = self.distances[valid_mask]
            log_j_valid = np.log(j_profile[valid_mask])

            poly_fit = np.polyfit(d_valid, log_j_valid, 1)
            alpha = -float(poly_fit[0])
            delta_fit = float(1.0 / (alpha + 1e-8)) if alpha > 0 else float("inf")
        else:
            alpha = 0.0
            delta_fit = float("inf")

        return {
            "delta_phase": float(delta_phase),
            "delta_phase_fit": float(delta_fit),
            "attenuation_coeff": float(alpha),
            "j_profile": j_profile,
            "distances": self.distances,
        }


def visualize_meta_topology(g_meta_tensor, step_info: str = "Final Step", show_plot: bool = False) -> plt.Figure:
    """
    g_meta 3D 텐서 [D, H, W]의 곡률 분포 및 단면 투영 시각화
    """
    if hasattr(g_meta_tensor, "cpu"):
        g_meta_data = g_meta_tensor.cpu().detach().numpy()
    else:
        g_meta_data = np.array(g_meta_tensor)

    depth, height, width = g_meta_data.shape

    fig = plt.figure(figsize=(14, 6))
    fig.suptitle(f"Meta-Topology Metric Curvature Field (g_meta) - {step_info}", fontsize=14, fontweight="bold")

    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    level_val = float(np.mean(g_meta_data) + np.std(g_meta_data) * 0.5)
    level_val = min(level_val, float(np.max(g_meta_data) - 0.01))

    if marching_cubes is not None and level_val > np.min(g_meta_data):
        try:
            verts, faces, normals, values = marching_cubes(g_meta_data, level=level_val)
            mesh = ax1.plot_trisurf(verts[:, 0], verts[:, 1], faces, verts[:, 2], cmap="magma", lw=0.1, alpha=0.85)
            ax1.set_title(f"3D Curvature Isosurface (g_meta = {level_val:.3f})")
        except Exception:
            x, y, z = np.indices((depth, height, width))
            sc = ax1.scatter(x, y, z, c=g_meta_data.flatten(), cmap="magma", alpha=0.3)
            ax1.set_title("3D Density Cloud Representation")
    else:
        x, y, z = np.indices((depth, height, width))
        sc = ax1.scatter(x, y, z, c=g_meta_data.flatten(), cmap="magma", alpha=0.3)
        ax1.set_title("3D Density Cloud Representation")

    ax1.set_xlabel("Depth (Z)")
    ax1.set_ylabel("Height (Y)")
    ax1.set_zlabel("Width (X)")

    ax2 = fig.add_subplot(1, 2, 2)
    mid_slice = g_meta_data[depth // 2, :, :]
    im = ax2.imshow(mid_slice, cmap="magma", origin="lower", interpolation="bicubic")
    ax2.set_title(f"Cross-Section Slice (Depth = {depth // 2})")
    ax2.set_xlabel("Width (X)")
    ax2.set_ylabel("Height (Y)")

    contours = ax2.contour(mid_slice, colors="white", alpha=0.4, linewidths=0.8)
    ax2.clabel(contours, inline=True, fontsize=8, fmt="%.2f")

    cbar = fig.colorbar(im, ax=[ax1, ax2], orientation="horizontal", pad=0.1, shrink=0.6)
    cbar.set_label("Metric Deformation Weight (g_meta)", fontsize=11)

    plt.tight_layout()
    if show_plot:
        plt.show()
    return fig


class TensorField3DVisualizer:
    """
    MultiVarAdaptiveOpenBoundaryPipeline의 내부 엔트로피 스칼라장 S(x,y,z) 및
    유입 플럭스 벡터장 J_flux(x,y,z)를 3D 공간 상에 매핑하여 렌더링하는 시각화기
    """

    def __init__(self, pipeline, grid_subsample: int = 2):
        self.pipeline = pipeline
        self.device = pipeline.q_sys.device
        self.shape = pipeline.shape
        self.dx = pipeline.dx
        self.subsample = grid_subsample

        d, h, w = self.shape
        z, y, x = np.meshgrid(
            np.arange(0, d, grid_subsample) * self.dx,
            np.arange(0, h, grid_subsample) * self.dx,
            np.arange(0, w, grid_subsample) * self.dx,
            indexing="ij",
        )
        self.grid_x, self.grid_y, self.grid_z = x, y, z
        self.sub_indices = (
            slice(0, d, grid_subsample),
            slice(0, h, grid_subsample),
            slice(0, w, grid_subsample),
        )

    @torch.no_grad()
    def compute_local_tensor_fields(self, q_ext: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """국소 엔트로피 스칼라 밀도 S(x,y,z) 및 플럭스 벡터 J_vec(x,y,z) 추출"""
        q_sys = self.pipeline.q_sys
        dq_dx = (torch.roll(q_sys, -1, 0) - q_sys) / self.dx
        dq_dy = (torch.roll(q_sys, -1, 1) - q_sys) / self.dx
        dq_dz = (torch.roll(q_sys, -1, 2) - q_sys) / self.dx

        grad_sq = torch.sum(dq_dx**2 + dq_dy**2 + dq_dz**2, dim=-1)
        prob_density = grad_sq / (torch.sum(grad_sq) + 1e-8)
        local_entropy = -prob_density * torch.log(prob_density + 1e-8)

        q_sys_inv = self.pipeline._quat_inv(q_sys)
        j_flux = self.pipeline._quat_mul(q_ext, q_sys_inv)
        j_vec = j_flux[..., 1:]

        S_field = local_entropy[self.sub_indices].cpu().numpy()
        J_vec_field = j_vec[self.sub_indices].cpu().numpy()

        return S_field, J_vec_field


class PyVistaTensorVolumeRenderer:
    """
    PyVista 볼륨 렌더링을 통한 3D 국소 위상 엔트로피 S(x,y,z) 스칼라장 및
    유입 플럭스 J_vec(x,y,z) 벡터장의 실시간 GPU 가속 시각화 모듈 (Optional)
    """

    def __init__(self, pipeline, dx: float = 0.1):
        if pv is None:
            raise ImportError("pyvista package is not installed.")
        self.pipeline = pipeline
        self.dx = dx
        self.shape = pipeline.shape

        self.grid = pv.ImageData()
        self.grid.dimensions = np.array(self.shape)
        self.grid.spacing = (dx, dx, dx)
        self.grid.origin = (0.0, 0.0, 0.0)

    @torch.no_grad()
    def _extract_tensor_fields(self, q_ext: torch.Tensor):
        q_sys = self.pipeline.q_sys
        dq_dx = (torch.roll(q_sys, -1, 0) - q_sys) / self.dx
        dq_dy = (torch.roll(q_sys, -1, 1) - q_sys) / self.dx
        dq_dz = (torch.roll(q_sys, -1, 2) - q_sys) / self.dx

        grad_sq = torch.sum(dq_dx**2 + dq_dy**2 + dq_dz**2, dim=-1)
        prob = grad_sq / (torch.sum(grad_sq) + 1e-8)
        local_entropy = -prob * torch.log(prob + 1e-8)

        q_sys_inv = self.pipeline._quat_inv(q_sys)
        j_flux = self.pipeline._quat_mul(q_ext, q_sys_inv)
        j_vec = j_flux[..., 1:]

        self.grid.point_data["entropy"] = local_entropy.cpu().numpy().flatten(order="F")
        self.grid.point_data["j_flux"] = j_vec.cpu().numpy().reshape(-1, 3, order="F")
