"""
Rotational Cyclic Morphology Module (로터적 순환 및 위상 형태학 모듈)

This module implements the fundamental wave dynamics where:
1. RotationalCyclicEngine: Uses complex phasors (A * e^(i*phi)) for continuous cyclic
   phase shift, intrinsic restoration forces, and energy/entropy conservation.
2. PhaseMorphologyField: Manages observation scale projections (Scale Frame of Reference)
   that transform geometric shapes (cubes, simplices, lines, curves, circles, spheres)
   as continuous morphing topology fields.
3. EnvelopeAreaBlock: Extracts amplitude/envelopes from discrete bit pulse intersections (0 & 1),
   filling the density space into continuous macro-unit geometric area blocks and integration
   with CausalField.
"""

import numpy as np
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, field

from core.physics.causal_field import CausalField, InformationVoxel, ConnectivityBeam


@dataclass
class PhasorState:
    """
    Complex Phasor state representing rotational wave dynamics.
    State: Z = A * exp(i * phi)
    """
    amplitude: float = 1.0
    phase: float = 0.0  # Phase angle in radians [0, 2*pi)
    frequency: float = 1.0  # Angular frequency w (rad/s)
    decay_rate: float = 0.0  # Natural damping / dissipation rate

    def to_complex(self) -> complex:
        return self.amplitude * np.exp(1j * self.phase)

    @classmethod
    def from_complex(cls, z: complex, frequency: float = 1.0) -> "PhasorState":
        amp = float(np.abs(z))
        phase = float(np.angle(z)) % (2.0 * np.pi)
        return cls(amplitude=amp, phase=phase, frequency=frequency)

    def rotate(self, dt: float) -> float:
        """
        Advances the phase angle based on frequency and dt.
        Returns the wrapped phase shift.
        """
        d_phase = self.frequency * dt
        self.phase = (self.phase + d_phase) % (2.0 * np.pi)
        if self.decay_rate > 0:
            self.amplitude *= np.exp(-self.decay_rate * dt)
        return d_phase


class RotationalCyclicEngine:
    """
    Rotational Cyclic Engine (로터적 순환 엔진)
    Replaces discrete scalar logic with complex phasor rotational dynamics (A * exp(i*phi)).
    Maintains intrinsic restoration forces (위상 복원력) and energy conservation tracking.
    """

    def __init__(self, default_frequency: float = 1.0, damping: float = 0.0):
        self.default_frequency = default_frequency
        self.damping = damping
        self.phasors: Dict[str, PhasorState] = {}
        self.initial_energies: Dict[str, float] = {}
        self.total_cycle_count: int = 0

    def register_phasor(
        self, phasor_id: str, amplitude: float = 1.0, phase: float = 0.0, frequency: Optional[float] = None
    ) -> PhasorState:
        freq = frequency if frequency is not None else self.default_frequency
        state = PhasorState(
            amplitude=amplitude,
            phase=phase % (2.0 * np.pi),
            frequency=freq,
            decay_rate=self.damping,
        )
        self.phasors[phasor_id] = state
        self.initial_energies[phasor_id] = self._calculate_energy(state)
        return state

    def _calculate_energy(self, state: PhasorState) -> float:
        # Energy E = 0.5 * A^2 * w^2 (Harmonic oscillator energy equivalence)
        return 0.5 * (state.amplitude ** 2) * (state.frequency ** 2)

    def apply_restoration_force(self, phasor_id: str, target_phase: float, coupling_k: float = 0.5, dt: float = 0.1):
        """
        Phase restoration force (위상 복원력):
        Pulls phasor phase smoothly towards target phase / natural phase equilibrium.
        """
        if phasor_id not in self.phasors:
            return
        state = self.phasors[phasor_id]
        phase_diff = np.arctan2(np.sin(target_phase - state.phase), np.cos(target_phase - state.phase))
        # Restoration torque
        restoration = coupling_k * phase_diff * dt
        state.phase = (state.phase + restoration) % (2.0 * np.pi)

    def step(self, dt: float = 0.1) -> Dict[str, Any]:
        """
        Evolves all registered phasors across cycle step.
        Returns cycle diagnostics including energy conservation ratio (E_conservation).
        """
        self.total_cycle_count += 1
        current_total_energy = 0.0
        initial_total_energy = sum(self.initial_energies.values()) + 1e-12

        states_summary = {}
        for pid, state in self.phasors.items():
            state.rotate(dt)
            energy = self._calculate_energy(state)
            current_total_energy += energy
            states_summary[pid] = {
                "amplitude": state.amplitude,
                "phase": state.phase,
                "real": float(np.real(state.to_complex())),
                "imag": float(np.imag(state.to_complex())),
                "energy": energy,
            }

        e_conservation = current_total_energy / initial_total_energy

        return {
            "cycle": self.total_cycle_count,
            "energy_conservation_ratio": float(e_conservation),
            "total_energy": current_total_energy,
            "phasors": states_summary,
        }


class PhaseMorphologyField:
    """
    Phase Morphology Field (위상 형태 장)
    Manages observation scales (Scale Frame of Reference) and projects underlying wavefields
    into morphing geometric forms:
    - Discrete Frame (Micro / Scale < 0.5): Cubes, Simplices, Discrete Lattices.
    - Continuous Transition Frame (Meso / 0.5 <= Scale <= 2.0): Smooth Lines, B-spline Curves.
    - Cosmic Wave Field Frame (Macro / Scale > 2.0): Rotational Circles, Spheres, Curl Vortices.
    """

    def __init__(self, observation_scale: float = 1.0):
        self.observation_scale = observation_scale

    def set_scale(self, scale: float):
        self.observation_scale = max(0.01, float(scale))

    def project_waveform_geometry(
        self, waveform_points: np.ndarray, phasors: Optional[List[PhasorState]] = None
    ) -> Dict[str, Any]:
        """
        Projects wave points based on observation_scale into morphing geometry.
        """
        pts = np.asarray(waveform_points, dtype=np.float32)
        n_points = len(pts)

        if phasors is None or len(phasors) == 0:
            phases = np.linspace(0, 2.0 * np.pi, n_points, endpoint=False)
            amps = np.ones(n_points)
        else:
            phases = np.array([p.phase for p in phasors[:n_points]])
            amps = np.array([p.amplitude for p in phasors[:n_points]])
            if len(phases) < n_points:
                phases = np.pad(phases, (0, n_points - len(phases)), mode="wrap")
                amps = np.pad(amps, (0, n_points - len(amps)), mode="edge")

        scale = self.observation_scale

        if scale < 0.5:
            # Discrete Lattice / Cube / Simplex projection
            morph_type = "DISCRETE_CUBE_SIMPLEX_LATTICE"
            quantized_pts = np.round(pts * (1.0 / scale)) * scale
            geometry = quantized_pts
        elif scale <= 2.0:
            # Continuous Morphing Line/Curve projection
            morph_type = "CONTINUOUS_LINE_CURVE_MORPH"
            wave_mod = amps[:, None] * np.column_stack([np.cos(phases), np.sin(phases), np.sin(phases * 0.5)])
            if wave_mod.shape[1] > pts.shape[1]:
                wave_mod = wave_mod[:, : pts.shape[1]]
            geometry = pts + wave_mod * (scale * 0.2)
        else:
            # Rotational Cyclic Sphere / Curl Vortex projection
            morph_type = "ROTATIONAL_CIRCLE_SPHERE_CURL"
            radius = scale * np.mean(amps)
            theta = phases
            phi = np.linspace(0, np.pi, n_points)
            x = radius * np.sin(phi) * np.cos(theta)
            y = radius * np.sin(phi) * np.sin(theta)
            z = radius * np.cos(phi)
            geometry = np.column_stack([x, y, z])

        # Topological invariant measure (e.g. centroid curvature / circulation)
        centroid = np.mean(geometry, axis=0)
        disp = geometry - centroid
        radii = np.linalg.norm(disp, axis=1)
        topological_invariant = float(np.std(radii) / (np.mean(radii) + 1e-6))

        return {
            "observation_scale": scale,
            "morph_type": morph_type,
            "projected_geometry": geometry,
            "centroid": centroid.tolist(),
            "topological_invariant": topological_invariant,
        }


class EnvelopeAreaBlock:
    """
    Envelope Area Block (포락선 면적 블록)
    Extracts amplitude and envelope from discrete 0 & 1 bit pulse crossings,
    filling (Rasterization / Area Filling) the interval into a macro-unit area block.
    Converts 0/1 bits from cold discrete points into a living 2D/3D shape block.
    """

    def __init__(self, block_id: str, grid_resolution: Tuple[int, int] = (16, 16)):
        self.block_id = block_id
        self.grid_resolution = grid_resolution

    def extract_envelope_and_fill(
        self, bit_stream: np.ndarray, amplitude_scale: float = 1.0
    ) -> Dict[str, Any]:
        """
        Takes binary bit sequence (0s and 1s), extracts smooth envelope via Hilbert/moving filter,
        and rasterizes/fills the envelope into a dense continuous area matrix (Area Block).
        """
        bits = np.asarray(bit_stream, dtype=np.float32).flatten()
        n_bits = len(bits)

        # 1. Envelope extraction from bit pulses
        # Smooth box convolution + Hilbert amplitude envelope analog
        kernel_size = max(3, n_bits // 8)
        kernel = np.ones(kernel_size) / kernel_size
        smooth_bits = np.convolve(bits, kernel, mode="same")

        # Upper and lower envelopes
        envelope_upper = smooth_bits * amplitude_scale

        # 2. Area Filling (Rasterization into 2D Grid)
        rows, cols = self.grid_resolution
        area_matrix = np.zeros((rows, cols), dtype=np.float32)

        x_indices = np.linspace(0, n_bits - 1, cols).astype(int)
        for c, x_idx in enumerate(x_indices):
            h_upper = envelope_upper[x_idx]
            max_row = int(np.clip(h_upper * rows, 0, rows))
            if max_row > 0:
                area_matrix[:max_row, c] = 1.0  # Dense area block fill

        # 3. Filling Ratio & Density Match calculation
        ideal_envelope_area = float(np.sum(envelope_upper) * (cols / n_bits) * rows)
        filled_matrix_area = float(np.sum(area_matrix))
        filling_ratio = (
            filled_matrix_area / ideal_envelope_area
            if ideal_envelope_area > 0
            else 1.0
        )
        density_match = float(np.clip(filling_ratio * 100.0, 0.0, 100.0))

        return {
            "block_id": self.block_id,
            "bit_count": n_bits,
            "envelope_upper": envelope_upper,
            "area_matrix": area_matrix,
            "filled_area": filled_matrix_area,
            "ideal_area": ideal_envelope_area,
            "filling_ratio": filling_ratio,
            "density_match_percent": density_match,
        }

    def convert_to_causal_voxels(
        self, envelope_data: Dict[str, Any], causal_field: CausalField
    ) -> List[InformationVoxel]:
        """
        Converts the filled continuous area block into CausalField InformationVoxels (Macro Units),
        upgrading cold 0/1 bits into living wavefield macro-units.
        """
        area_matrix = envelope_data["area_matrix"]
        rows, cols = area_matrix.shape
        created_voxels = []

        active_indices = np.argwhere(area_matrix > 0.1)
        for idx, (r, c) in enumerate(active_indices):
            v_id = f"{self.block_id}_voxel_{r}_{c}"
            pos = np.array([c / cols, r / rows, 0.0], dtype=np.float32)
            density_val = area_matrix[r, c]

            voxel = InformationVoxel(
                id=v_id,
                content=f"MacroUnitAreaBlock_{self.block_id}",
                tensor=np.array([density_val, c / cols, r / rows], dtype=np.float32),
                mass=float(density_val),
                position=pos,
                phase_angle=float((c / cols) * 2.0 * np.pi),
            )
            causal_field.add_voxel(voxel)
            created_voxels.append(voxel)

        # Connect adjacent voxels to form a structural connectivity beam block
        for i in range(len(created_voxels) - 1):
            v_a = created_voxels[i]
            v_b = created_voxels[i + 1]
            if np.linalg.norm(v_a.position - v_b.position) < 0.2:
                causal_field.link_voxels(v_a.id, v_b.id, strength=1.5)

        return created_voxels
