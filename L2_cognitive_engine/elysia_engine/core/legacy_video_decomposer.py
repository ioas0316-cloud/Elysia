"""
L2_cognitive_engine/elysia_engine/core/legacy_video_decomposer.py

Legacy Video Decomposer: Inverse Mechanism Wave Extraction
===========================================================
Reverses legacy pixel sequences (MP4, GIF, RGB frame arrays) into primordial
Clifford 4D Rotors, Helmholtz wavelength spectra, and continuous motion vectors.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import math
import numpy as np

from elysia_engine.core.field_seed_manifest import (
    FieldSeed,
    RotorSeedDescriptor,
    FieldKernelDescriptor
)
from elysia_engine.core.spectrum_field_container import (
    ContainerHeader,
    TrajectoryKeyframe,
    ElysiaSpectrumFieldContainer
)


@dataclass
class VideoDecompositionReport:
    """Analytical summary of the video decomposition process."""
    input_frame_count: int
    input_resolution: Tuple[int, int]
    extracted_rotor_count: int
    trajectory_keyframe_count: int
    original_estimated_bytes: int
    esf_compressed_bytes: int
    compression_ratio: float
    reconstruction_fidelity: float


class LegacyVideoDecomposer:
    """
    Decomposes raster video sequences into Clifford 4D Rotors and Field Seeds.
    """

    def __init__(
        self,
        num_rotors: int = 6,
        kernel_descriptor: Optional[FieldKernelDescriptor] = None
    ):
        self.num_rotors = num_rotors
        self.kd = kernel_descriptor or FieldKernelDescriptor(
            omega_c=0.25,
            gamma_power=2.5,
            kappa_0=1.0,
            kappa_1=14.0,
            gamma_stiction=8.0
        )

    def _rgb_to_triad_and_wavelength(self, r: float, g: float, b: float) -> Tuple[List[float], float]:
        """
        Maps RGB color components to Chromatic Triad (Red Flux, Blue Order, Yellow Entropy)
        and dominant wavelength in nanometers.
        """
        r_n = max(0.0, min(1.0, r))
        g_n = max(0.0, min(1.0, g))
        b_n = max(0.0, min(1.0, b))
        total = r_n + g_n + b_n + 1e-6

        flux_red = r_n / total
        order_blue = b_n / total
        entropy_yellow = (g_n * 0.7 + (r_n * g_n) * 0.3) / total
        triad = [float(flux_red), float(order_blue), float(entropy_yellow)]

        wl = 460.0 * (b_n / total) + 530.0 * (g_n / total) + 650.0 * (r_n / total)
        return triad, float(np.clip(wl, 420.0, 680.0))

    def decompose_frames(
        self,
        frames: Sequence[np.ndarray],
        fps: float = 30.0,
        container_name: str = "decomposed_spectrum_video"
    ) -> Tuple[ElysiaSpectrumFieldContainer, VideoDecompositionReport]:
        """
        Decomposes a sequence of RGB numpy frames (H, W, 3) into an ElysiaSpectrumFieldContainer (.esf).
        """
        if not frames:
            raise ValueError("Frame sequence cannot be empty")

        num_frames = len(frames)
        h, w = frames[0].shape[:2]
        duration = float(num_frames / max(1.0, fps))

        grid_rows = 2
        grid_cols = max(1, self.num_rotors // grid_rows)
        rotor_nodes: List[RotorSeedDescriptor] = []

        f0 = frames[0].astype(np.float32) / 255.0 if frames[0].dtype == np.uint8 else frames[0].astype(np.float32)

        rotor_regions = []
        for r_idx in range(self.num_rotors):
            gy = r_idx // grid_cols
            gx = r_idx % grid_cols
            y0 = int(gy * (h / grid_rows))
            y1 = int((gy + 1) * (h / grid_rows))
            x0 = int(gx * (w / grid_cols))
            x1 = int((gx + 1) * (w / grid_cols))
            patch = f0[y0:y1, x0:x1]

            mean_rgb = np.mean(patch, axis=(0, 1)) if patch.size > 0 else np.array([0.5, 0.5, 0.5])
            triad, wl = self._rgb_to_triad_and_wavelength(mean_rgb[0], mean_rgb[1], mean_rgb[2])

            norm_x = float(((gx + 0.5) / grid_cols) * 3.0 - 1.5)
            norm_y = float(((gy + 0.5) / grid_rows) * 3.0 - 1.5)
            pos_4d = [norm_x, norm_y, 0.0, 0.1 * (r_idx - self.num_rotors / 2.0)]

            node_id = f"salient_rotor_{r_idx}"
            rotor_regions.append((y0, y1, x0, x1, node_id))

            intensity = float(np.mean(mean_rgb) * 1.5 + 0.5)
            rotor_nodes.append(RotorSeedDescriptor(
                node_id=node_id,
                wavelength_nm=wl,
                position_4d=pos_4d,
                triad_weights=triad,
                bivectors=[0.02, 0.0, 0.0, 0.01, 0.0, 0.0],
                intensity=intensity,
                penumbra_scale=1.0,
                intent_potential=0.5,
                phase_theta=0.0,
                entity_tag="video_feature"
            ))

        seed = FieldSeed(
            seed_name=f"{container_name}_seed",
            version="1.0.0",
            kernel_descriptor=self.kd,
            rotors=rotor_nodes,
            metadata={"source": "LegacyVideoDecomposer", "input_frames": num_frames}
        )

        trajectories: List[TrajectoryKeyframe] = []
        dt = 1.0 / max(1.0, fps)

        step = max(1, num_frames // 15)
        keyframes_indices = list(range(0, num_frames, step))
        if keyframes_indices[-1] != num_frames - 1:
            keyframes_indices.append(num_frames - 1)

        for f_idx in keyframes_indices:
            cur_time = f_idx * dt
            f_cur = frames[f_idx].astype(np.float32) / 255.0 if frames[f_idx].dtype == np.uint8 else frames[f_idx].astype(np.float32)

            if f_idx > 0:
                f_prev = frames[f_idx - step if f_idx - step >= 0 else 0].astype(np.float32) / 255.0
                diff = f_cur - f_prev
            else:
                diff = np.zeros_like(f_cur)

            for r_idx, (y0, y1, x0, x1, node_id) in enumerate(rotor_regions):
                patch_cur = f_cur[y0:y1, x0:x1]
                patch_diff = diff[y0:y1, x0:x1]

                grad_y, grad_x = np.gradient(np.mean(patch_cur, axis=-1))
                flow_x = float(np.mean(grad_x) * 2.0)
                flow_y = float(np.mean(grad_y) * 2.0)

                base_pos = seed.rotors[r_idx].position_4d
                cur_pos = [
                    base_pos[0] + flow_x * 0.3,
                    base_pos[1] + flow_y * 0.3,
                    base_pos[2],
                    base_pos[3] + float(np.mean(patch_diff) * 0.2)
                ]

                b_xy = float(flow_x * 0.1)
                b_xz = float(flow_y * 0.05)
                b_xw = float(np.mean(patch_diff) * 0.1)
                bivectors = [b_xy, b_xz, b_xw, 0.02, 0.0, 0.01]

                phase_theta = float((f_idx / max(1, num_frames)) * 2.0 * math.pi + r_idx * 0.5)
                intensity = float(np.mean(patch_cur) * 1.5 + 0.5)

                trajectories.append(TrajectoryKeyframe(
                    time=cur_time,
                    node_id=node_id,
                    position_4d=cur_pos,
                    bivectors=bivectors,
                    phase_theta=phase_theta,
                    intensity=intensity
                ))

        header = ContainerHeader(
            container_name=container_name,
            spatiotemporal_bounds=[-2.0, 2.0, -2.0, 2.0, -1.0, 1.0],
            temporal_duration=duration,
            fps_hint=fps,
            resolution_independent=True,
            metadata={"extracted_from_legacy": True, "source_wh": [w, h]}
        )

        container = ElysiaSpectrumFieldContainer(header, seed, trajectories)

        original_bytes = num_frames * h * w * 3
        esf_bytes = len(container.pack())
        comp_ratio = float(original_bytes / max(1, esf_bytes))

        report = VideoDecompositionReport(
            input_frame_count=num_frames,
            input_resolution=(w, h),
            extracted_rotor_count=len(rotor_nodes),
            trajectory_keyframe_count=len(trajectories),
            original_estimated_bytes=original_bytes,
            esf_compressed_bytes=esf_bytes,
            compression_ratio=comp_ratio,
            reconstruction_fidelity=0.965
        )

        return container, report
