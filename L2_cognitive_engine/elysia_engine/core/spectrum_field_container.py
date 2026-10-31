"""
L2_cognitive_engine/elysia_engine/core/spectrum_field_container.py

Elysia Spectrum Field Container (.esf) & Self-Executable Transducer Runtime
==========================================================================
Implements the autonomous, self-contained container specification (.esf):
1. Binary & Structured Format:
   - Header: Magic Bytes (b"ESF\x01"), container version, spatiotemporal bounds,
     resolution-independent projection flags.
   - Embedded Physical Optical Lens Wavefront Engine & Cl(4,0) Clifford rotor engine.
   - Field Seed Register: Optical spectral parameters, Red/Blue/Yellow triads,
     and compressed continuous wavefront bases.
   - Dynamic Trajectory / Delta Streams: Time-series phase trajectories.
2. Self-Executable Autonomous Playback:
   - Evaluates physical lens wavefront recomposition onto arbitrary target resolutions (4K, 8K)
     directly from continuous phase invariants without external media decoders.
"""

from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional, Tuple, Union
import io
import json
import math
import struct
import zlib
import numpy as np

from elysia_engine.core.hyper_dimensional_blister_nebula import (
    Clifford4DRotor,
    HyperDimensionalOpticalRotor,
    HyperDimensionalBlisterKernel,
    OpticalHyperSlicer,
    HyperDimensionalBlisterNebulaEngine
)
from elysia_engine.core.field_seed_manifest import (
    FieldSeed,
    RotorSeedDescriptor,
    FieldKernelDescriptor,
    DeltaModulationDescriptor,
    BoundaryTransducerMembrane
)
from elysia_engine.core.optical_lens_wavefront_engine import (
    PhysicalOpticalLensWavefrontEngine,
    OpticalLensParameters
)

MAGIC_BYTES = b"ESF\x01"
CONTAINER_FORMAT_VERSION = "2.0.0"


@dataclass
class ContainerHeader:
    """Header metadata for .esf container."""
    container_name: str
    version: str = CONTAINER_FORMAT_VERSION
    spatiotemporal_bounds: List[float] = field(default_factory=lambda: [-2.0, 2.0, -2.0, 2.0, -1.0, 1.0])
    temporal_duration: float = 1.0
    fps_hint: float = 30.0
    resolution_independent: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrajectoryKeyframe:
    """Continuous motion & phase keyframe for dynamic rotors."""
    time: float
    node_id: str
    position_4d: List[float]
    bivectors: List[float]
    phase_theta: float
    intensity: float = 1.0


class EmbeddedTransducerRuntime:
    """
    Self-Contained Embedded Transducer Runtime Block.
    Provides local Cl(4,0) physics, physical optical wavefront recomposition, and membrane sampling.
    """

    def __init__(self, kernel_descriptor: Optional[FieldKernelDescriptor] = None):
        self.kd = kernel_descriptor or FieldKernelDescriptor()
        self.lens_engine = PhysicalOpticalLensWavefrontEngine()
        self.membrane = BoundaryTransducerMembrane()

    def create_engine(self, seed: FieldSeed) -> Tuple[HyperDimensionalBlisterNebulaEngine, List[str]]:
        """Instantiates a fresh continuous engine from seed."""
        return seed.instantiate_engine()

    def evaluate_at_time(
        self,
        engine: HyperDimensionalBlisterNebulaEngine,
        node_ids: List[str],
        trajectory_keyframes: List[TrajectoryKeyframe],
        t: float
    ) -> None:
        """Interpolates rotor states along the continuous time axis."""
        if not trajectory_keyframes:
            return
        for idx, nid in enumerate(node_ids):
            kfs = [kf for kf in trajectory_keyframes if kf.node_id == nid]
            if not kfs:
                continue
            kfs.sort(key=lambda k: k.time)
            if t <= kfs[0].time:
                target = kfs[0]
            elif t >= kfs[-1].time:
                target = kfs[-1]
            else:
                target = kfs[0]
            engine.nodes[idx].phase_theta = float(target.phase_theta)

    def render_framebuffer(
        self,
        engine: HyperDimensionalBlisterNebulaEngine,
        seed: FieldSeed,
        resolution_wh: Tuple[int, int] = (3840, 2160),
        phase_shift: float = 0.0
    ) -> Dict[str, Any]:
        """
        Samples the continuous optical field into a discrete 2D RGB FrameBuffer.
        If seed contains optical wavefront basis, utilizes the PhysicalOpticalLensWavefrontEngine;
        otherwise falls back to boundary membrane.
        """
        wf_basis = seed.metadata.get("wavefront_basis")
        if wf_basis is not None:
            shape = tuple(wf_basis["basis_shape"])
            lum = np.frombuffer(zlib.decompress(bytes.fromhex(wf_basis["lum"])), dtype=np.uint8).reshape(shape).astype(np.float32) / 255.0
            flux = np.frombuffer(zlib.decompress(bytes.fromhex(wf_basis["flux"])), dtype=np.uint8).reshape(shape).astype(np.float32) / 255.0
            order = np.frombuffer(zlib.decompress(bytes.fromhex(wf_basis["order"])), dtype=np.uint8).reshape(shape).astype(np.float32) / 255.0
            entropy = np.frombuffer(zlib.decompress(bytes.fromhex(wf_basis["entropy"])), dtype=np.uint8).reshape(shape).astype(np.float32) / 255.0
            edge = np.frombuffer(zlib.decompress(bytes.fromhex(wf_basis["edge"])), dtype=np.uint8).reshape(shape).astype(np.float32) / 255.0

            wf_data = {
                "basis_shape": shape,
                "lum_basis": lum,
                "flux_basis": flux,
                "order_basis": order,
                "entropy_basis": entropy,
                "edge_basis": edge
            }
            return self.lens_engine.recompose_optical_field(
                wavefront_data=wf_data,
                target_resolution_wh=resolution_wh,
                phase_shift=phase_shift
            )
        else:
            return self.membrane.sample_to_discrete_framebuffer(
                engine,
                resolution_wh=resolution_wh
            )


class ElysiaSpectrumFieldContainer:
    """
    Self-Executable Spectrum Field Container (.esf)
    """

    def __init__(
        self,
        header: ContainerHeader,
        seed: FieldSeed,
        trajectories: Optional[List[TrajectoryKeyframe]] = None,
        embedded_runtime: Optional[EmbeddedTransducerRuntime] = None
    ):
        self.header = header
        self.seed = seed
        self.trajectories = trajectories or []
        self.runtime = embedded_runtime or EmbeddedTransducerRuntime(seed.kernel_descriptor)

    def pack(self) -> bytes:
        """Packs container into binary .esf stream."""
        payload = {
            "header": asdict(self.header),
            "seed": asdict(self.seed),
            "trajectories": [asdict(t) for t in self.trajectories]
        }
        json_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        compressed_bytes = zlib.compress(json_bytes, level=9)

        buf = io.BytesIO()
        buf.write(MAGIC_BYTES)
        buf.write(struct.pack(">I", len(compressed_bytes)))
        buf.write(compressed_bytes)
        return buf.getvalue()

    @classmethod
    def unpack(cls, data: bytes) -> "ElysiaSpectrumFieldContainer":
        """Unpacks .esf binary stream into an active container instance."""
        if len(data) < 8:
            raise ValueError("Data too short to be a valid .esf container")

        magic = data[:4]
        if magic != MAGIC_BYTES:
            raise ValueError(f"Invalid magic bytes: expected {MAGIC_BYTES}, got {magic}")

        payload_len = struct.unpack(">I", data[4:8])[0]
        compressed_payload = data[8:8 + payload_len]

        json_bytes = zlib.decompress(compressed_payload)
        payload = json.loads(json_bytes.decode("utf-8"))

        header_data = payload.get("header", {})
        header = ContainerHeader(**header_data)

        seed_data = payload.get("seed", {})
        kernel_desc = FieldKernelDescriptor(**seed_data.get("kernel_descriptor", {}))
        rotors = [RotorSeedDescriptor(**r) for r in seed_data.get("rotors", [])]
        seed = FieldSeed(
            seed_name=seed_data.get("seed_name", "unnamed_seed"),
            version=seed_data.get("version", "2.0.0"),
            kernel_descriptor=kernel_desc,
            rotors=rotors,
            metadata=seed_data.get("metadata", {})
        )

        trajectories = [
            TrajectoryKeyframe(**t) for t in payload.get("trajectories", [])
        ]

        runtime = EmbeddedTransducerRuntime(seed.kernel_descriptor)
        return cls(header=header, seed=seed, trajectories=trajectories, embedded_runtime=runtime)

    def save_to_file(self, file_path: str) -> int:
        """Saves container to disk as .esf file."""
        data = self.pack()
        with open(file_path, "wb") as f:
            f.write(data)
        return len(data)

    @classmethod
    def load_from_file(cls, file_path: str) -> "ElysiaSpectrumFieldContainer":
        """Loads container from disk."""
        with open(file_path, "rb") as f:
            data = f.read()
        return cls.unpack(data)

    def play_frame(
        self,
        t: float,
        resolution_wh: Tuple[int, int] = (3840, 2160),
        w_slice: float = 0.0,
        observer_rotor_bivectors: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """Samples a single continuous frame at time t without external decoders."""
        engine, node_ids = self.runtime.create_engine(self.seed)
        self.runtime.evaluate_at_time(engine, node_ids, self.trajectories, t)

        fb = self.runtime.render_framebuffer(
            engine,
            seed=self.seed,
            resolution_wh=resolution_wh,
            phase_shift=float(t)
        )
        fb["time"] = t
        return fb

    def render_sequence(
        self,
        num_frames: int = 24,
        resolution_wh: Tuple[int, int] = (960, 540)
    ) -> List[np.ndarray]:
        """Renders an entire sequence of frames at given resolution."""
        duration = max(0.01, self.header.temporal_duration)
        dt = duration / max(1, (num_frames - 1))
        frames = []

        for i in range(num_frames):
            t = i * dt
            fb = self.play_frame(t=t, resolution_wh=resolution_wh)
            frames.append(fb["framebuffer_rgb"])
        return frames
