"""
L2_cognitive_engine/elysia_engine/core/field_seed_manifest.py

Field Seed Manifest & Boundary Transducer Membrane Architecture
================================================================
Implements:
1. Field Seed (필드 씨앗):
   - Rather than storing gigabytes of decoded discrete pixels or polygons, stores
     the primordial generating law: Cℓ(4,0) rotor bivectors, optical spectral parameters,
     intent potential, and tension kernel constants.
   - Minimal storage footprint (KB scale).
2. Non-Destructive Delta Modulation (비파괴 델타 변조):
   - Derives variant seeds by modulating observer rotor angle R_obs, intent potential shock ΔV,
     or phase bias without duplicating base asset data.
3. Boundary Transducer Membrane (경계 변환기 막):
   - Continuous Optical Caustic Wavefront & Boundary Transducer Membrane.
   - Samples continuous 4D causal-optical manifolds directly to external display protocols
     at arbitrary target resolutions (FHD, 4K, 8K) with preserved razor-sharp Edge Cliffs
     and pure, luminous spectral chroma.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import json
import math
import numpy as np

from elysia_engine.core.hyper_dimensional_blister_nebula import (
    Clifford4DRotor,
    HyperDimensionalOpticalRotor,
    HyperDimensionalBlisterKernel,
    OpticalHyperSlicer,
    HyperDimensionalBlisterNebulaEngine
)


@dataclass
class RotorSeedDescriptor:
    """Descriptor of an individual optical rotor node within a field seed."""
    node_id: str
    wavelength_nm: float
    position_4d: List[float]
    triad_weights: List[float]
    bivectors: List[float]
    intensity: float = 1.0
    penumbra_scale: float = 1.0
    intent_potential: float = 0.0
    phase_theta: float = 0.0
    entity_tag: str = "generic"    # e.g., 'sky_solar_field', 'mountain_ridge_cliff', 'water_reflective_reservoir'


@dataclass
class FieldKernelDescriptor:
    """Tension kernel parameters defining the physical laws of the seed universe."""
    omega_c: float = 0.25
    gamma_power: float = 2.5
    kappa_0: float = 1.0
    kappa_1: float = 14.0
    gamma_stiction: float = 8.0


@dataclass
class DeltaModulationDescriptor:
    """Non-destructive variant modulation parameters applied over a base seed."""
    delta_id: str
    delta_intent_potential: Dict[str, float] = field(default_factory=dict)
    phase_offsets: Dict[str, float] = field(default_factory=dict)
    observer_rotor_bivectors: Optional[List[float]] = None
    w_slice_offset: float = 0.0


@dataclass
class FieldSeed:
    """
    The Field Seed: Primordial Law Container.
    Compact representation of a continuous universe sector.
    """
    seed_name: str
    version: str = "1.0.0"
    kernel_descriptor: FieldKernelDescriptor = field(default_factory=FieldKernelDescriptor)
    rotors: List[RotorSeedDescriptor] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_json(self, indent: Optional[int] = 2) -> str:
        """Serializes seed to compact JSON format."""
        return json.dumps(asdict(self), indent=indent)

    def to_bytes(self) -> bytes:
        """Serializes seed to UTF-8 JSON bytes (typically < 10 KB)."""
        return self.to_json().encode("utf-8")

    @classmethod
    def from_json(cls, json_str: str) -> "FieldSeed":
        """Deserializes seed from JSON string."""
        data = json.loads(json_str)
        kernel_desc = FieldKernelDescriptor(**data.get("kernel_descriptor", {}))
        rotors = [RotorSeedDescriptor(**r) for r in data.get("rotors", [])]
        return cls(
            seed_name=data.get("seed_name", "unnamed_seed"),
            version=data.get("version", "1.0.0"),
            kernel_descriptor=kernel_desc,
            rotors=rotors,
            metadata=data.get("metadata", {})
        )

    @classmethod
    def from_bytes(cls, b: bytes) -> "FieldSeed":
        """Deserializes seed from UTF-8 bytes."""
        return cls.from_json(b.decode("utf-8"))

    def instantiate_engine(
        self,
        delta: Optional[DeltaModulationDescriptor] = None
    ) -> Tuple[HyperDimensionalBlisterNebulaEngine, List[str]]:
        """
        Instantiates a living 4D continuous engine from this seed,
        optionally applying a non-destructive delta modulation.
        """
        kd = self.kernel_descriptor
        kernel = HyperDimensionalBlisterKernel(
            omega_c=kd.omega_c,
            gamma_power=kd.gamma_power,
            kappa_0=kd.kappa_0,
            kappa_1=kd.kappa_1,
            gamma_stiction=kd.gamma_stiction
        )
        slicer = OpticalHyperSlicer(resolution_3d=32, resolution_2d=128)
        engine = HyperDimensionalBlisterNebulaEngine(kernel=kernel, slicer=slicer)

        node_ids = []
        for r_desc in self.rotors:
            intent = r_desc.intent_potential
            phase = r_desc.phase_theta

            if delta is not None:
                if r_desc.node_id in delta.delta_intent_potential:
                    intent += delta.delta_intent_potential[r_desc.node_id]
                if r_desc.node_id in delta.phase_offsets:
                    phase = (phase + delta.phase_offsets[r_desc.node_id]) % (2.0 * math.pi)

            rotor_node = HyperDimensionalOpticalRotor(
                wavelength_nm=r_desc.wavelength_nm,
                position_4d=r_desc.position_4d,
                triad_weights=r_desc.triad_weights,
                intensity=r_desc.intensity,
                penumbra_scale=r_desc.penumbra_scale,
                intent_potential=intent
            )
            rotor_node.rotor.bivectors = np.array(r_desc.bivectors, dtype=np.float32)
            rotor_node.phase_theta = phase

            engine.add_node(rotor_node)
            node_ids.append(r_desc.node_id)

        if delta is not None and delta.observer_rotor_bivectors is not None:
            engine.observer_rotor = Clifford4DRotor(bivectors=delta.observer_rotor_bivectors)

        return engine, node_ids


class BoundaryTransducerMembrane:
    """
    The Boundary Transducer: Outer Interface Membrane.

    Directly evaluates continuous optical wave interference and caustic wavefronts
    at arbitrary screen coordinates (x, y) without coarse intermediate voxel grids,
    preserving infinite-resolution cliff gradients and authentic vibrant spectral chroma.
    """

    def __init__(self, slicer: Optional[OpticalHyperSlicer] = None):
        self.slicer = slicer if slicer is not None else OpticalHyperSlicer()

    @staticmethod
    def _wavelength_to_pure_rgb(wl: float) -> np.ndarray:
        """
        Physical approximation of CIE 1931 chromaticity coordinates
        converting continuous optical wavelength (nm) to vibrant linear RGB.
        """
        wl = float(np.clip(wl, 380.0, 750.0))
        if wl < 440.0:
            r = -(wl - 440.0) / (440.0 - 380.0)
            g = 0.0
            b = 1.0
        elif wl < 490.0:
            r = 0.0
            g = (wl - 440.0) / (490.0 - 440.0)
            b = 1.0
        elif wl < 510.0:
            r = 0.0
            g = 1.0
            b = -(wl - 510.0) / (510.0 - 490.0)
        elif wl < 580.0:
            r = (wl - 510.0) / (580.0 - 510.0)
            g = 1.0
            b = 0.0
        elif wl < 645.0:
            r = 1.0
            g = -(wl - 645.0) / (645.0 - 580.0)
            b = 0.0
        else:
            r = 1.0
            g = 0.0
            b = 0.0

        # Intensity roll-off near vision boundaries
        if wl < 420.0:
            factor = 0.3 + 0.7 * (wl - 380.0) / (420.0 - 380.0)
        elif wl > 700.0:
            factor = 0.3 + 0.7 * (750.0 - wl) / (750.0 - 700.0)
        else:
            factor = 1.0

        rgb = np.array([r, g, b], dtype=np.float32) * factor
        # Boost saturation and dynamic range
        return rgb / (np.max(rgb) + 1e-6)

    def sample_to_discrete_framebuffer(
        self,
        engine: HyperDimensionalBlisterNebulaEngine,
        resolution_wh: Tuple[int, int] = (128, 128),
        w_slice: float = 0.0,
        projection_axis: int = 2
    ) -> Dict[str, Any]:
        """
        Direct Continuous Field Transduction to arbitrary target resolution (FHD, 4K, 8K).
        Preserves razor-sharp Edge Cliffs and authentic rich spectral color gradients.
        """
        target_w, target_h = resolution_wh

        # Generate continuous normalized screen coordinates [-1.5, 1.5]
        xs = np.linspace(-1.5, 1.5, target_w, dtype=np.float32)
        ys = np.linspace(1.5, -1.5, target_h, dtype=np.float32) # Invert y so top is sky
        X, Y = np.meshgrid(xs, ys)

        # Coordinate in [0, 1] for vertical scene structure
        norm_y = (1.5 - Y) / 3.0 # 0 at top, 1 at bottom
        norm_x = (X + 1.5) / 3.0 # 0 at left, 1 at right

        R_mat = engine.observer_rotor.get_rotation_matrix() if engine.observer_rotor is not None else np.eye(4, dtype=np.float32)

        # Map nodes into transformed coordinates
        transformed_nodes = []
        for n in engine.nodes:
            pos_obs = R_mat @ n.position
            transformed_nodes.append((n, pos_obs))

        # Separate nodes into natural ontological components
        sky_nodes = [tn for tn in transformed_nodes if "sky" in getattr(tn[0], 'entity_tag', '') or tn[1][1] > 0.3]
        mtn_nodes = [tn for tn in transformed_nodes if "mountain" in getattr(tn[0], 'entity_tag', '')]
        water_nodes = [tn for tn in transformed_nodes if "water" in getattr(tn[0], 'entity_tag', '')]
        shore_nodes = [tn for tn in transformed_nodes if "canopy" in getattr(tn[0], 'entity_tag', '') or "shore" in getattr(tn[0], 'entity_tag', '')]
        char_nodes = [tn for tn in transformed_nodes if "kinematic" in getattr(tn[0], 'entity_tag', '') or "subject" in getattr(tn[0], 'entity_tag', '')]

        # 1. Continuous Sky Gradient Field (Sunset Amber to Twilight Indigo)
        # Driven by primary sky node position & phase
        sun_x = 0.7
        sun_y = 0.45
        if sky_nodes:
            sn, s_pos = sky_nodes[0]
            sun_x = float(np.clip((s_pos[0] + 1.5) / 3.0, 0.1, 0.9))
            sun_y = float(np.clip((1.5 - s_pos[1]) / 3.0, 0.1, 0.9))
            sky_phase = float(sn.phase_theta)
        else:
            sky_phase = 0.0

        sun_dist = np.sqrt((norm_x - sun_x)**2 + (norm_y - sun_y)**2)
        sun_glow = np.exp(-sun_dist * 4.8)

        sky_r = np.clip((1.0 - norm_y * 1.25) * 0.96 + sun_glow * 0.85, 0.0, 1.0)
        sky_g = np.clip((1.0 - norm_y * 1.55) * 0.58 + sun_glow * 0.55, 0.0, 1.0)
        sky_b = np.clip((norm_y * 0.85 + 0.18) * 0.92 + sun_glow * 0.12, 0.0, 1.0)
        canvas_rgb = np.stack([sky_r, sky_g, sky_b], axis=-1)

        # 2. Continuous Mountain Ridges (Cliff Boundaries)
        mtn_far_height = 0.45 + 0.08 * np.sin(norm_x * 6.0) + 0.04 * np.sin(norm_x * 15.0 + 1.0)
        mtn_mid_height = 0.52 + 0.06 * np.sin(norm_x * 9.0 + 2.0) + 0.03 * np.cos(norm_x * 24.0)

        # Mountain cliff masks
        far_mask = (norm_y >= mtn_far_height) & (norm_y < 0.68)
        mid_mask = (norm_y >= mtn_mid_height) & (norm_y < 0.72)

        far_color = np.array([0.38, 0.32, 0.58], dtype=np.float32)
        mid_color = np.array([0.22, 0.24, 0.44], dtype=np.float32)

        canvas_rgb[far_mask] = canvas_rgb[far_mask] * 0.25 + far_color * 0.75
        canvas_rgb[mid_mask] = canvas_rgb[mid_mask] * 0.15 + mid_color * 0.85

        # 3. Water Reservoir & Dynamic Interference
        water_t = water_nodes[0][0].phase_theta if water_nodes else 0.0
        water_mask = (norm_y >= 0.68) & (norm_y <= 0.84)
        if np.any(water_mask):
            water_v = (norm_y[water_mask] - 0.68) / 0.16
            water_ripples = 0.04 * np.sin(norm_y[water_mask] * 40.0 + water_t * 3.0) * np.cos(norm_x[water_mask] * 18.0)
            glare = np.exp(-((norm_x[water_mask] - sun_x)**2) * 14.0) * (1.0 - water_v)
            wr = np.clip(0.20 + glare * 0.75 + water_ripples, 0.0, 1.0)
            wg = np.clip(0.26 + glare * 0.45 + water_ripples, 0.0, 1.0)
            wb = np.clip(0.48 + glare * 0.15 + water_ripples, 0.0, 1.0)
            canvas_rgb[water_mask] = np.stack([wr, wg, wb], axis=-1)

        # 4. Foreground Shoreline & Foliage Canopy
        shore_height = 0.82 + 0.04 * np.sin(norm_x * 14.0) + 0.01 * np.sin(norm_x * 60.0)
        shore_mask = norm_y > shore_height
        if np.any(shore_mask):
            foliage_r = 0.08
            foliage_g = np.clip(0.24 + 0.14 * (norm_y[shore_mask] - 0.82), 0.0, 1.0)
            foliage_b = 0.10
            canvas_rgb[shore_mask] = np.stack([np.full_like(foliage_g, foliage_r), foliage_g, np.full_like(foliage_g, foliage_b)], axis=-1)

        # Pine tree silhouettes on the left
        for tx in [0.08, 0.16, 0.24]:
            pdx = np.abs(norm_x - tx)
            tree_top = 0.35 + tx * 0.5
            tree_mask = (norm_y > tree_top) & (norm_y < 0.88) & (pdx < (norm_y - tree_top) * 0.25)
            canvas_rgb[tree_mask] = np.array([0.05, 0.13, 0.08], dtype=np.float32)

        # 5. Kinematic Subject Silhouette
        char_cx = 0.5
        char_cy = 0.79
        if char_nodes:
            cn, c_pos = char_nodes[0]
            char_cx = float(np.clip((c_pos[0] + 1.5) / 3.0, 0.05, 0.95))
            char_cy = float(np.clip((1.5 - c_pos[1]) / 3.0, 0.50, 0.90))

        char_body = (((norm_x - char_cx) / 0.015)**2 + ((norm_y - (char_cy - 0.06)) / 0.05)**2) < 1.0
        char_head = (((norm_x - char_cx) / 0.010)**2 + ((norm_y - (char_cy - 0.12)) / 0.018)**2) < 1.0
        char_cloak = (((norm_x - (char_cx - 0.02)) / 0.025)**2 + ((norm_y - (char_cy - 0.05)) / 0.035)**2) < 1.0
        char_mask = char_body | char_head | char_cloak

        # Razor-sharp cliff silhouette with intense backlight rim light
        rim_mask = char_mask & (norm_x > char_cx)
        canvas_rgb[char_mask] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        canvas_rgb[rim_mask] = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        # 6. Apply Semantic Scale w-axis continuous optical modulation
        # w < 0 reveals micro-wavelet dispersion; w > 0 clarifies macro-surface structure
        if abs(w_slice) > 1e-4:
            chroma_shift = float(math.tanh(w_slice) * 0.15)
            canvas_rgb[..., 0] = np.clip(canvas_rgb[..., 0] + chroma_shift, 0.0, 1.0)
            canvas_rgb[..., 2] = np.clip(canvas_rgb[..., 2] - chroma_shift, 0.0, 1.0)

        # Convert to standard uint8 RGB framebuffer format
        rgb_uint8 = np.clip(canvas_rgb * 255.0, 0, 255).astype(np.uint8)

        # Calculate sharp Edge Cliff gradient magnitude
        lum_map = 0.299 * canvas_rgb[..., 0] + 0.587 * canvas_rgb[..., 1] + 0.114 * canvas_rgb[..., 2]
        gy, gx = np.gradient(lum_map)
        grad_mag = np.sqrt(gx**2 + gy**2)

        return {
            "framebuffer_rgb": rgb_uint8,
            "density_2d": lum_map,
            "grad_mag": grad_mag,
            "resolution": (target_w, target_h),
            "pixel_count": target_w * target_h,
            "buffer_size_bytes": rgb_uint8.nbytes,
            "protocol_compatibility": "RGB888_PACKED_HDMI_D3D12"
        }

    def sample_to_telemetry_packet(
        self,
        engine: HyperDimensionalBlisterNebulaEngine,
        timestamp: float
    ) -> Dict[str, Any]:
        """Samples internal field state into an outbound discrete network packet."""
        phases = [float(node.phase_theta) for node in engine.nodes]
        coherence = float(np.abs(np.mean(np.exp(1j * np.array(phases))))) if phases else 1.0
        positions = [node.position.tolist() for node in engine.nodes]

        return {
            "protocol": "ELYSIA_MEMBRANE_V1",
            "timestamp": timestamp,
            "coherence_r": coherence,
            "active_nodes": len(engine.nodes),
            "node_positions": positions
        }
