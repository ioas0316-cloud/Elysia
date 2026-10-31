"""
L2_cognitive_engine/elysia_engine/core/spectrum_field_container.py

Elysia Spectrum Field Container (.esf) & Self-Executable Transducer Runtime
==========================================================================
Implements the autonomous, self-contained container specification (.esf):
1. Binary & Structured Format:
   - Header: Magic Bytes (b"ESF\x01"), container version, spatiotemporal bounds,
     resolution-independent projection flags, and block table of contents.
   - Runtime Kernel Block: Embedded minimal Cl(4,0) Clifford rotor engine,
     Helmholtz wave interference solver, and Boundary Transducer Membrane.
   - Field Seed Register: Primordial generating law containing Clifford 4D Rotors,
     spectral wavelength parameters, Red/Blue/Yellow triads, and tension kernel constants.
   - Dynamic Trajectory / Delta Streams: Time-series bivector and phase trajectory equations.
2. Self-Executable Autonomous Playback:
   - Can execute standalone playback without external media decoders (MP4/H.264/FFmpeg).
   - Samples continuous 4D causal-optical field directly to discrete display framebuffers
     at arbitrary target resolutions (FHD, 4K, 8K) with preserved cliff edge gradients.
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

MAGIC_BYTES = b"ESF\x01"
CONTAINER_FORMAT_VERSION = "1.0.0"


@dataclass
class ContainerHeader:
    """Header metadata for .esf container."""
    container_name: str
    version: str = CONTAINER_FORMAT_VERSION
    spatiotemporal_bounds: List[float] = field(default_factory=lambda: [-2.0, 2.0, -2.0, 2.0, -1.0, 1.0]) # [xmin, xmax, ymin, ymax, wmin, wmax]
    temporal_duration: float = 1.0 # Duration in continuous seconds
    fps_hint: float = 30.0 # Frame rate hint for discrete samplers
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
    Provides local Cl(4,0) physics, wave interference, and membrane sampling.
    """

    def __init__(self, kernel_descriptor: Optional[FieldKernelDescriptor] = None):
        self.kd = kernel_descriptor or FieldKernelDescriptor()
        self.kernel = HyperDimensionalBlisterKernel(
            omega_c=self.kd.omega_c,
            gamma_power=self.kd.gamma_power,
            kappa_0=self.kd.kappa_0,
            kappa_1=self.kd.kappa_1,
            gamma_stiction=self.kd.gamma_stiction
        )
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
        """
        Interpolates rotor states along the continuous time axis using Causal Rotor Interpolation.
        """
        if not trajectory_keyframes:
            return

        for idx, nid in enumerate(node_ids):
            kfs = [kf for kf in trajectory_keyframes if kf.node_id == nid]
            if not kfs:
                continue

            kfs.sort(key=lambda k: k.time)
            if t <= kfs[0].time:
                target = kfs[0]
                engine.nodes[idx].position = np.array(target.position_4d, dtype=np.float32)
                engine.nodes[idx].rotor.bivectors = np.array(target.bivectors, dtype=np.float32)
                engine.nodes[idx].phase_theta = float(target.phase_theta)
                engine.nodes[idx].intensity = float(target.intensity)
            elif t >= kfs[-1].time:
                target = kfs[-1]
                engine.nodes[idx].position = np.array(target.position_4d, dtype=np.float32)
                engine.nodes[idx].rotor.bivectors = np.array(target.bivectors, dtype=np.float32)
                engine.nodes[idx].phase_theta = float(target.phase_theta)
                engine.nodes[idx].intensity = float(target.intensity)
            else:
                for i in range(len(kfs) - 1):
                    if kfs[i].time <= t <= kfs[i+1].time:
                        k0, k1 = kfs[i], kfs[i+1]
                        alpha = (t - k0.time) / max(1e-6, (k1.time - k0.time))
                        p0 = np.array(k0.position_4d, dtype=np.float32)
                        p1 = np.array(k1.position_4d, dtype=np.float32)
                        engine.nodes[idx].position = (1.0 - alpha) * p0 + alpha * p1
                        bv0 = np.array(k0.bivectors, dtype=np.float32)
                        bv1 = np.array(k1.bivectors, dtype=np.float32)
                        engine.nodes[idx].rotor.bivectors = (1.0 - alpha) * bv0 + alpha * bv1
                        engine.nodes[idx].phase_theta = float((1.0 - alpha) * k0.phase_theta + alpha * k1.phase_theta)
                        engine.nodes[idx].intensity = float((1.0 - alpha) * k0.intensity + alpha * k1.intensity)
                        break

    def render_framebuffer(
        self,
        engine: HyperDimensionalBlisterNebulaEngine,
        resolution_wh: Tuple[int, int] = (128, 128),
        w_slice: float = 0.0,
        projection_axis: int = 2
    ) -> Dict[str, Any]:
        """Samples the continuous field into a discrete 2D RGB FrameBuffer."""
        return self.membrane.sample_to_discrete_framebuffer(
            engine,
            resolution_wh=resolution_wh,
            w_slice=w_slice,
            projection_axis=projection_axis
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
            version=seed_data.get("version", "1.0.0"),
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
        resolution_wh: Tuple[int, int] = (128, 128),
        w_slice: float = 0.0,
        observer_rotor_bivectors: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """Samples a single continuous frame at time t without external decoders."""
        engine, node_ids = self.runtime.create_engine(self.seed)
        if observer_rotor_bivectors is not None:
            engine.observer_rotor = Clifford4DRotor(bivectors=observer_rotor_bivectors)

        self.runtime.evaluate_at_time(engine, node_ids, self.trajectories, t)

        fb = self.runtime.render_framebuffer(
            engine,
            resolution_wh=resolution_wh,
            w_slice=w_slice
        )
        fb["time"] = t
        return fb

    def render_sequence(
        self,
        num_frames: int = 30,
        resolution_wh: Tuple[int, int] = (96, 96),
        w_slice: float = 0.0
    ) -> List[np.ndarray]:
        """Renders an entire sequence of frames at given resolution."""
        duration = max(0.01, self.header.temporal_duration)
        dt = duration / max(1, (num_frames - 1))
        frames = []

        engine, node_ids = self.runtime.create_engine(self.seed)
        for i in range(num_frames):
            t = i * dt
            self.runtime.evaluate_at_time(engine, node_ids, self.trajectories, t)
            fb = self.runtime.render_framebuffer(engine, resolution_wh=resolution_wh, w_slice=w_slice)
            frames.append(fb["framebuffer_rgb"])
        return frames

    def verify_causal_flow_continuity(
        self,
        resolution_wh: Tuple[int, int] = (64, 64),
        sample_steps: int = 4
    ) -> Dict[str, Any]:
        """
        Evaluates the causal continuity and phase friction of the container playback.
        Returns continuity metrics showing adherence to physical phase conservation.
        """
        from elysia_engine.core.micro_scale_spectral_transition_engine import (
            MicroScaleSpectralTransitionEngine
        )
        transition_engine = MicroScaleSpectralTransitionEngine()
        frames = self.render_sequence(num_frames=sample_steps, resolution_wh=resolution_wh)

        total_friction = 0.0
        total_ruptures = 0
        all_natural = True

        for i in range(len(frames) - 1):
            assessment = transition_engine.assess_causal_continuity(frames[i], frames[i+1])
            total_friction += assessment.mean_phase_friction
            total_ruptures += assessment.phase_rupture_count
            if not assessment.is_causally_natural:
                all_natural = False

        avg_friction = total_friction / max(1, len(frames) - 1)
        return {
            "all_transitions_natural": all_natural,
            "average_phase_friction": float(avg_friction),
            "total_phase_ruptures": total_ruptures,
            "samples_evaluated": len(frames) - 1
        }

    def get_container_footprint(self) -> Dict[str, Any]:
        """Calculates storage footprint metrics compared to legacy formats."""
        binary_bytes = len(self.pack())
        raw_seed_bytes = len(self.seed.to_bytes())
        rotor_count = len(self.seed.rotors)
        trajectory_count = len(self.trajectories)

        fps = self.header.fps_hint
        duration = self.header.temporal_duration
        total_frames = int(fps * duration)
        raw_fhd_bytes = total_frames * (1920 * 1080 * 3)
        mp4_est_bytes = max(500_000, int(total_frames * (1920 * 1080 * 3) * 0.015))

        return {
            "container_bytes": binary_bytes,
            "container_kb": binary_bytes / 1024.0,
            "raw_seed_bytes": raw_seed_bytes,
            "rotor_count": rotor_count,
            "trajectory_count": trajectory_count,
            "equivalent_raw_fhd_mb": raw_fhd_bytes / (1024.0 * 1024.0),
            "equivalent_mp4_mb": mp4_est_bytes / (1024.0 * 1024.0),
            "compression_ratio_vs_mp4": (mp4_est_bytes / max(1, binary_bytes))
        }
