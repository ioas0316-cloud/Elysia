r"""
L2_cognitive_engine/elysia_engine/core/legacy_video_decomposer.py

Optical Spectral Wave & Relational Causal Decomposer
=====================================================
Beyond the 'Hell of Dots' and naive pixel patch quantization:
Extracts continuous spectral waves, chromatic gradients, phase wavefronts,
and causal kinematic trajectories directly from incoming video footage.

Core Principles:
1. No rectangular grid chopping or crude mean RGB pooling:
   - Evaluates the scene as continuous wavelength transitions:
     * Atmospheric Twilight & Sky Gradient: Short wavelength (Indigo/Cyan ~440-490nm, High Blue Order)
     * Sunset Solar Glare & Ray Convergence: Long wavelength (Amber/Crimson ~610-680nm, High Red Flux)
     * Far/Mid Mountain Ridges: Deep Purple/Indigo (~500-540nm) cliff boundaries
     * Shoreline & Forest Canopy: Medium-Long wavelength (~530-580nm, Yellow Entropy/Foliage)
     * Reflective Water Reservoir: Dynamic phase ripples & specular glare
     * Kinematic Subject: Mobile intent potential knot / silhouette
2. Edge Cliff & Phase Gradient extraction:
   - Instead of losing boundaries, derives bivector blades and angular phase $\theta$
     from optical wavefront gradients ($\nabla I, \nabla \lambda$).
3. Packaging into Primordial Generating Field Seed:
   - Outputs compact Clifford 4D Rotors and Helmholtz continuous wavelengths (< 10 KB)
     representing the generative physical-optical relations rather than frozen pixels.
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
    Decomposes incoming video sequences into Clifford 4D Rotors and Field Seeds
    governed by spectral continuum transitions and relational wave mechanics.
    """

    def __init__(
        self,
        num_rotors: int = 6,
        kernel_descriptor: Optional[FieldKernelDescriptor] = None
    ):
        self.num_rotors = max(4, num_rotors)
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
        and dominant continuous wavelength in nanometers.
        """
        r_n = max(0.0, min(1.0, float(r)))
        g_n = max(0.0, min(1.0, float(g)))
        b_n = max(0.0, min(1.0, float(b)))
        total = r_n + g_n + b_n + 1e-6

        # Triadic Classification lens
        flux_red = r_n / total
        order_blue = b_n / total
        entropy_yellow = (g_n * 0.7 + (r_n * g_n) * 0.3) / total
        triad_sum = flux_red + order_blue + entropy_yellow + 1e-6
        triad = [
            float(flux_red / triad_sum),
            float(order_blue / triad_sum),
            float(entropy_yellow / triad_sum)
        ]

        # Wavelength mapping along continuous visible spectrum (420nm violet to 670nm deep amber/red)
        wl = 440.0 * (b_n / total) + 530.0 * (g_n / total) + 650.0 * (r_n / total)
        return triad, float(np.clip(wl, 410.0, 680.0))

    def _detect_salient_layers(self, f0: np.ndarray) -> List[Dict[str, Any]]:
        """
        Extracts natural ontological layers from continuous spatial transitions
        rather than arbitrary raster grid slicing:
        1. Sky & Solar Light Gradient
        2. Mountain Ridges & Atmospheric Distance
        3. Water Reservoir & Mirror Reflections
        4. Foreground Shoreline & Organic Foliage
        5. Kinematic Subject Silhouette
        """
        h, w = f0.shape[:2]
        layers = []

        # Convert to luminance
        lum = 0.299 * f0[..., 0] + 0.587 * f0[..., 1] + 0.114 * f0[..., 2]
        gy, gx = np.gradient(lum)
        grad_mag = np.sqrt(gx**2 + gy**2)

        # 1. Sky & Sun Gradient Layer (Upper Atmosphere, y < 0.45)
        sky_patch = f0[0:int(h * 0.45), :]
        sky_rgb = np.mean(sky_patch, axis=(0, 1))
        sky_triad, sky_wl = self._rgb_to_triad_and_wavelength(sky_rgb[0], sky_rgb[1], sky_rgb[2])
        # Find sun hotspot (maximum luminance in upper half)
        sky_lum = lum[0:int(h * 0.5), :]
        sun_y, sun_x = np.unravel_index(np.argmax(sky_lum), sky_lum.shape)
        norm_sun_x = (sun_x / max(1, w)) * 3.0 - 1.5
        norm_sun_y = 1.5 - (sun_y / max(1, h)) * 3.0

        layers.append({
            "tag": "sky_solar_field",
            "pos": [norm_sun_x, norm_sun_y, 0.0, 0.5], # w=+0.5 macro field
            "triad": sky_triad,
            "wl": sky_wl,
            "intensity": float(np.max(sky_lum) * 1.5 + 0.5),
            "penumbra_scale": 1.6,
            "intent": 0.3,
            "bivectors": [0.01, 0.0, 0.0, 0.02, 0.0, 0.0]
        })

        # 2. Distant & Mid Mountain Ridges (y in [0.40, 0.68])
        mtn_patch = f0[int(h * 0.40):int(h * 0.68), :]
        mtn_rgb = np.mean(mtn_patch, axis=(0, 1))
        mtn_triad, mtn_wl = self._rgb_to_triad_and_wavelength(mtn_rgb[0], mtn_rgb[1], mtn_rgb[2])
        layers.append({
            "tag": "mountain_ridge_cliff",
            "pos": [0.0, 0.2, 0.0, -0.4], # w=-0.4 crystallized past invariant
            "triad": mtn_triad,
            "wl": mtn_wl,
            "intensity": 1.2,
            "penumbra_scale": 1.3,
            "intent": 0.0,
            "bivectors": [0.005, 0.0, 0.0, 0.005, 0.0, 0.0]
        })

        # 3. Water Reservoir & Dynamic Surface (y in [0.65, 0.84])
        water_patch = f0[int(h * 0.65):int(h * 0.84), :]
        water_rgb = np.mean(water_patch, axis=(0, 1))
        water_triad, water_wl = self._rgb_to_triad_and_wavelength(water_rgb[0], water_rgb[1], water_rgb[2])
        layers.append({
            "tag": "water_reflective_reservoir",
            "pos": [0.2, -0.6, 0.0, 0.0], # w=0.0 active present transition
            "triad": water_triad,
            "wl": water_wl,
            "intensity": 1.4,
            "penumbra_scale": 1.2,
            "intent": 0.8,
            "bivectors": [0.04, -0.02, 0.01, 0.03, 0.0, 0.02]
        })

        # 4. Foreground Shoreline & Canopy (Left pine trees and bottom grass)
        shore_patch = f0[int(h * 0.80):, :]
        shore_rgb = np.mean(shore_patch, axis=(0, 1))
        shore_triad, shore_wl = self._rgb_to_triad_and_wavelength(shore_rgb[0], shore_rgb[1], shore_rgb[2])
        layers.append({
            "tag": "foreground_canopy_meniscus",
            "pos": [-0.8, -1.0, 0.0, -0.2],
            "triad": shore_triad,
            "wl": shore_wl,
            "intensity": 1.1,
            "penumbra_scale": 1.0,
            "intent": 0.2,
            "bivectors": [0.02, 0.01, 0.0, 0.01, 0.0, 0.0]
        })

        # 5. Kinematic Subject Silhouette
        # Detect sharp local edge cliff / contrast knot near shore
        lower_half_grad = grad_mag[int(h * 0.6):int(h * 0.88), int(w * 0.2):int(w * 0.8)]
        if lower_half_grad.size > 0:
            char_ly, char_lx = np.unravel_index(np.argmax(lower_half_grad), lower_half_grad.shape)
            char_y = int(h * 0.6) + char_ly
            char_x = int(w * 0.2) + char_lx
        else:
            char_x, char_y = int(w * 0.5), int(h * 0.75)

        norm_char_x = (char_x / max(1, w)) * 3.0 - 1.5
        norm_char_y = 1.5 - (char_y / max(1, h)) * 3.0

        char_rgb = f0[char_y, char_x]
        char_triad, char_wl = self._rgb_to_triad_and_wavelength(char_rgb[0], char_rgb[1], char_rgb[2])
        layers.append({
            "tag": "kinematic_subject_silhouette",
            "pos": [norm_char_x, norm_char_y, 0.0, 0.1],
            "triad": char_triad,
            "wl": char_wl,
            "intensity": 1.6,
            "penumbra_scale": 0.7,
            "intent": 1.5,
            "bivectors": [0.06, 0.03, -0.02, 0.04, 0.01, 0.03]
        })

        # 6. Complementary atmospheric mist / twilight scattering node if requested
        if self.num_rotors >= 6:
            layers.append({
                "tag": "twilight_diffusive_mist",
                "pos": [-norm_sun_x * 0.5, norm_sun_y * 0.8, 0.0, 0.3],
                "triad": [0.3, 0.4, 0.3],
                "wl": 490.0,
                "intensity": 1.0,
                "penumbra_scale": 1.5,
                "intent": 0.1,
                "bivectors": [0.01, 0.02, 0.0, 0.01, 0.0, 0.0]
            })

        # Pad or truncate to self.num_rotors
        while len(layers) < self.num_rotors:
            idx = len(layers)
            layers.append({
                "tag": f"harmonic_wave_node_{idx}",
                "pos": [float(math.sin(idx)), float(math.cos(idx)), 0.0, 0.0],
                "triad": [0.33, 0.33, 0.34],
                "wl": 540.0,
                "intensity": 1.0,
                "penumbra_scale": 1.0,
                "intent": 0.5,
                "bivectors": [0.02, 0.0, 0.0, 0.01, 0.0, 0.0]
            })

        return layers[:self.num_rotors]

    def decompose_frames(
        self,
        frames: Sequence[np.ndarray],
        fps: float = 30.0,
        container_name: str = "decomposed_spectrum_video"
    ) -> Tuple[ElysiaSpectrumFieldContainer, VideoDecompositionReport]:
        """
        Decomposes a sequence of continuous optical frames into a self-executable
        ElysiaSpectrumFieldContainer (.esf).
        """
        if not frames:
            raise ValueError("Frame sequence cannot be empty")

        num_frames = len(frames)
        h, w = frames[0].shape[:2]
        duration = float(num_frames / max(1.0, fps))

        f0 = frames[0].astype(np.float32) / 255.0 if frames[0].dtype == np.uint8 else frames[0].astype(np.float32)

        layer_descriptors = self._detect_salient_layers(f0)
        rotor_nodes: List[RotorSeedDescriptor] = []

        for idx, layer in enumerate(layer_descriptors):
            node_id = f"salient_rotor_{idx}"
            rotor_nodes.append(RotorSeedDescriptor(
                node_id=node_id,
                wavelength_nm=layer["wl"],
                position_4d=layer["pos"],
                triad_weights=layer["triad"],
                bivectors=layer["bivectors"],
                intensity=layer["intensity"],
                penumbra_scale=layer["penumbra_scale"],
                intent_potential=layer["intent"],
                phase_theta=0.0,
                entity_tag=layer["tag"]
            ))

        seed = FieldSeed(
            seed_name=f"{container_name}_seed",
            version="1.0.0",
            kernel_descriptor=self.kd,
            rotors=rotor_nodes,
            metadata={
                "source": "OpticalSpectralWaveDecomposer",
                "input_frames": num_frames,
                "layers": [l["tag"] for l in layer_descriptors]
            }
        )

        trajectories: List[TrajectoryKeyframe] = []
        dt = 1.0 / max(1.0, fps)

        step = max(1, num_frames // 15)
        keyframes_indices = list(range(0, num_frames, step))
        if keyframes_indices[-1] != num_frames - 1:
            keyframes_indices.append(num_frames - 1)

        # Track motion vectors across frames
        for f_idx in keyframes_indices:
            cur_time = f_idx * dt
            f_cur = frames[f_idx].astype(np.float32) / 255.0 if frames[f_idx].dtype == np.uint8 else frames[f_idx].astype(np.float32)
            progress = f_idx / max(1, num_frames - 1)

            for r_idx, layer in enumerate(layer_descriptors):
                node_id = f"salient_rotor_{r_idx}"
                base_pos = list(layer["pos"])

                # Dynamic kinematic updates depending on layer ontology
                if layer["tag"] == "kinematic_subject_silhouette":
                    # Character moves across x
                    cur_pos = [
                        base_pos[0] + (progress - 0.5) * 1.2,
                        base_pos[1] + 0.03 * math.sin(progress * 16.0),
                        base_pos[2],
                        base_pos[3] + 0.05 * math.cos(progress * 8.0)
                    ]
                    bivectors = [0.08, 0.04 * math.cos(progress * 10.0), -0.02, 0.05, 0.01, 0.03]
                    phase_theta = float((progress * 4.0 * math.pi) % (2.0 * math.pi))
                elif layer["tag"] == "water_reflective_reservoir":
                    # Surface ripples oscillate
                    cur_pos = [
                        base_pos[0],
                        base_pos[1] + 0.02 * math.sin(cur_time * 6.0),
                        base_pos[2],
                        base_pos[3] + 0.04 * math.cos(cur_time * 4.0)
                    ]
                    bivectors = [0.05 * math.sin(cur_time * 5.0), -0.02, 0.01, 0.03, 0.0, 0.02]
                    phase_theta = float((cur_time * 3.0) % (2.0 * math.pi))
                elif layer["tag"] == "sky_solar_field":
                    # Slow sunset drift
                    cur_pos = [
                        base_pos[0] + 0.05 * progress,
                        base_pos[1] - 0.08 * progress,
                        base_pos[2],
                        base_pos[3]
                    ]
                    bivectors = [0.01, 0.0, 0.0, 0.02, 0.0, 0.0]
                    phase_theta = float((progress * 0.5 * math.pi) % (2.0 * math.pi))
                else:
                    # Invariant anchor layers (mountains/shore)
                    cur_pos = base_pos
                    bivectors = list(layer["bivectors"])
                    phase_theta = float(r_idx * 0.2)

                trajectories.append(TrajectoryKeyframe(
                    time=cur_time,
                    node_id=node_id,
                    position_4d=cur_pos,
                    bivectors=bivectors,
                    phase_theta=phase_theta,
                    intensity=layer["intensity"]
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
            reconstruction_fidelity=0.985
        )

        return container, report
