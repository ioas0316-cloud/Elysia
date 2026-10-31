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
   - Acts as the outermost biological/technological interface membrane.
   - Keeps pure continuous 4D causal-optical manifolds sovereign inside, while
     transducing/sampling on-demand onto external legacy protocols:
     * Discrete Display FrameBuffer (RGB/RGBA pixels for HDMI/DisplayPort/Vulkan)
     * Discrete Network Stream (packets for WebSocket/WebRTC)
   - Enables arbitrary resolution sampling (FHD, 4K, 16K) from the same seed without DLSS.
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
    entity_tag: str = "generic"    # e.g., 'person', 'building', 'forest', 'light_source'


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
            # Base parameters
            intent = r_desc.intent_potential
            phase = r_desc.phase_theta

            # Apply non-destructive delta if present
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

        # Apply observer rotor from delta if specified
        if delta is not None and delta.observer_rotor_bivectors is not None:
            engine.observer_rotor = Clifford4DRotor(bivectors=delta.observer_rotor_bivectors)

        return engine, node_ids


class BoundaryTransducerMembrane:
    """
    The Boundary Transducer: Outer Interface Membrane.

    Interfaces the sovereign continuous high-dimensional field with the
    discrete legacy protocol reality (HDMI/Framebuffer, RGB pixels, network packets).
    """

    def __init__(self, slicer: Optional[OpticalHyperSlicer] = None):
        self.slicer = slicer if slicer is not None else OpticalHyperSlicer()

    def sample_to_discrete_framebuffer(
        self,
        engine: HyperDimensionalBlisterNebulaEngine,
        resolution_wh: Tuple[int, int] = (128, 128),
        w_slice: float = 0.0,
        projection_axis: int = 2
    ) -> Dict[str, Any]:
        """
        Samples the continuous 4D manifold into a standard 2D RGB FrameBuffer
        for legacy display protocols (HDMI, DisplayPort, Vulkan Framebuffer).
        """
        target_w, target_h = resolution_wh
        res_3d = min(40, max(24, int(max(target_w, target_h) / 3)))
        custom_slicer = OpticalHyperSlicer(resolution_3d=res_3d, resolution_2d=max(target_w, target_h))

        vol = custom_slicer.slice_4d_to_3d_volume(
            engine.nodes,
            engine.kernel,
            w_slice=w_slice,
            rotor_obs=engine.observer_rotor
        )
        proj = custom_slicer.project_to_2d_prism(vol, projection_axis=projection_axis)

        # Convert to standard uint8 RGB framebuffer format
        rgb_float = proj["rgb_2d"]

        # Resample or crop to exact target (target_h, target_w)
        if rgb_float.shape[0] != target_h or rgb_float.shape[1] != target_w:
            from scipy.ndimage import zoom
            zoom_y = target_h / rgb_float.shape[0]
            zoom_x = target_w / rgb_float.shape[1]
            rgb_float = zoom(rgb_float, (zoom_y, zoom_x, 1.0), order=1)

        rgb_uint8 = np.clip(rgb_float * 255.0, 0, 255).astype(np.uint8)

        # Compute metadata for external legacy protocols
        buffer_size_bytes = rgb_uint8.nbytes
        pixel_count = rgb_uint8.shape[0] * rgb_uint8.shape[1]

        return {
            "framebuffer_rgb": rgb_uint8,
            "density_2d": proj["density_2d"],
            "grad_mag": proj["grad_mag"],
            "resolution": (rgb_uint8.shape[1], rgb_uint8.shape[0]),
            "pixel_count": pixel_count,
            "buffer_size_bytes": buffer_size_bytes,
            "protocol_compatibility": "RGB888_PACKED_HDMI_D3D12"
        }

    def sample_to_telemetry_packet(
        self,
        engine: HyperDimensionalBlisterNebulaEngine,
        timestamp: float
    ) -> Dict[str, Any]:
        """
        Samples internal field state into an outbound discrete network packet
        (e.g., JSON/UDP/WebRTC signaling).
        """
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
