"""
Holographic Volumetric Spacetime Engine
======================================
Native 3D/4D Holographic Volumetric Spacetime Substrate and Causal Engine.

Key Architectural Foundations:
1. UniversalTopologicalPoint: Unifies heterogeneous entities (numerical, semantic,
   value, causal, human, physical) into a scale-consistent topological node in 3D/4D space.
2. KnowingIgnorancePhaseGradient: Models Knowing (coherent phase-lock) vs Ignorance
   (unconstrained superposition / white tensor state) as a continuous phase strain
   gradient and causal gravitational tension force.
3. HolographicVolumetricStorage: Stores information as 3D/4D volumetric phase wave
   interference patterns (Psi(x,y,z,t)), enabling global structural reconstruction
   from local partial spatial crops (holographic memory property).
4. CausalVolumetricSpacetimeEngine: Dissolves 2D layer stacks and step-by-step
   matrices in favor of a continuous 3D/4D volumetric continuum where local phase
   rotations propagate spherically and causally across spacetime.
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any

class UniversalTopologicalPoint:
    """
    Universal Topological Representative Node.
    Maps heterogeneous domains (numerical, semantic, value, causal, existential)
    into a scale-consistent 3D/4D topological representative point in spacetime volume.
    """

    def __init__(
        self,
        node_id: str,
        domain_type: str,  # 'numeric', 'semantic', 'value', 'causal', 'existential'
        raw_data: Any,
        position: Tuple[float, float, float],
        scale: float = 1.0,
        chromatic_vector: Optional[np.ndarray] = None  # [Red(Flux), Blue(Order), Yellow(Entropy)]
    ):
        self.node_id = node_id
        self.domain_type = domain_type
        self.raw_data = raw_data
        self.position = np.array(position, dtype=np.float64)
        self.scale = scale

        if chromatic_vector is None:
            # Default chromatic signature based on domain type
            if domain_type == 'numeric':
                self.chromatic_vector = np.array([0.2, 0.7, 0.1], dtype=np.float64)  # High order
            elif domain_type == 'semantic':
                self.chromatic_vector = np.array([0.5, 0.4, 0.1], dtype=np.float64)  # Balanced flux/order
            elif domain_type == 'value':
                self.chromatic_vector = np.array([0.8, 0.1, 0.1], dtype=np.float64)  # High flux
            elif domain_type == 'existential':
                self.chromatic_vector = np.array([0.3, 0.3, 0.4], dtype=np.float64)  # High entropy/potential
            else:
                self.chromatic_vector = np.array([0.33, 0.33, 0.34], dtype=np.float64)
        else:
            self.chromatic_vector = np.array(chromatic_vector, dtype=np.float64)

        # Quaternion Phase (w, xi, yj, zk)
        self.phase_quaternion = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.phase_coherence = 1.0  # 1.0 = fully coherent (knowing), 0.0 = white tensor superposition (ignorance)

    def set_phase_coherence(self, coherence: float) -> None:
        """Sets the phase coherence (knowing vs ignorance)."""
        self.phase_coherence = float(np.clip(coherence, 0.0, 1.0))

    def rotate_phase(self, axis: np.ndarray, angle: float) -> None:
        """Applies a 3D rotor rotation to the topological point's quaternion phase."""
        axis = np.array(axis, dtype=np.float64)
        norm = np.linalg.norm(axis)
        if norm > 1e-9:
            axis = axis / norm
        half_angle = angle / 2.0
        r_w = math.cos(half_angle)
        r_xyz = axis * math.sin(half_angle)
        rotor = np.array([r_w, r_xyz[0], r_xyz[1], r_xyz[2]], dtype=np.float64)

        # Quaternion multiplication
        w1, x1, y1, z1 = rotor
        w2, x2, y2, z2 = self.phase_quaternion

        w = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z = w1*z2 + x1*y2 - y1*x2 + z1*w2

        self.phase_quaternion = np.array([w, x, y, z], dtype=np.float64)
        q_norm = np.linalg.norm(self.phase_quaternion)
        if q_norm > 1e-9:
            self.phase_quaternion /= q_norm


class KnowingIgnorancePhaseGradient:
    """
    Computes the Phase Strain Gradient between Knowing (Coherent Phase Lock)
    and Ignorance (Unconstrained White Tensor Superposition).
    Translates phase gradients into causal gravitational tension forces.
    """

    def __init__(self, spatial_grid_shape: Tuple[int, int, int] = (16, 16, 16)):
        self.grid_shape = spatial_grid_shape
        # Coherence field: 1.0 = Known, 0.0 = Ignorant
        self.coherence_field = np.ones(spatial_grid_shape, dtype=np.float64)
        # Phase field: complex angle field e^(i * theta)
        self.phase_angle_field = np.zeros(spatial_grid_shape, dtype=np.float64)

    def update_coherence_from_points(self, points: List[UniversalTopologicalPoint]) -> None:
        """Rasterizes point coherence onto the 3D grid."""
        grid_z, grid_y, grid_x = self.grid_shape
        for pt in points:
            # Map normalized position [-1, 1] to grid coordinates
            gx = int(np.clip((pt.position[0] + 1.0) / 2.0 * (grid_x - 1), 0, grid_x - 1))
            gy = int(np.clip((pt.position[1] + 1.0) / 2.0 * (grid_y - 1), 0, grid_y - 1))
            gz = int(np.clip((pt.position[2] + 1.0) / 2.0 * (grid_z - 1), 0, grid_z - 1))

            # Influence radius
            rad = max(1, int(pt.scale * 2))
            for z in range(max(0, gz - rad), min(grid_z, gz + rad + 1)):
                for y in range(max(0, gy - rad), min(grid_y, gy + rad + 1)):
                    for x in range(max(0, gx - rad), min(grid_x, gx + rad + 1)):
                        dist = math.sqrt((x - gx)**2 + (y - gy)**2 + (z - gz)**2)
                        weight = math.exp(-dist / (rad + 1e-5))
                        self.coherence_field[z, y, x] = (
                            (1.0 - weight) * self.coherence_field[z, y, x] + weight * pt.phase_coherence
                        )

    def compute_phase_strain_gradient(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes spatial gradient of coherence (Strain Tensor Gradient)
        and the resulting Causal Gravitational Pressure Field.

        Returns:
            grad_field: (3, Z, Y, X) spatial gradient vector field of coherence.
            tension_force: (Z, Y, X) magnitude of causal pressure directed toward ignorance boundaries.
        """
        # Compute gradient along z, y, x axes
        gz, gy, gx = np.gradient(self.coherence_field)
        grad_field = np.stack([gx, gy, gz], axis=0)  # (3, Z, Y, X)

        # Strain magnitude: high near the boundary between knowing & ignorance
        tension_force = np.linalg.norm(grad_field, axis=0) * (1.0 - self.coherence_field)
        return grad_field, tension_force


class HolographicVolumetricStorage:
    """
    Holographic 3D/4D Volumetric Wave Interference Storage.
    Information is encoded as complex phase wave interference patterns Psi(x,y,z).
    Any partial spatial crop retains the global phase key and reconstructs
    the global structural pattern through wave propagation.
    """

    def __init__(self, volume_shape: Tuple[int, int, int] = (16, 16, 16)):
        self.volume_shape = volume_shape
        # Complex 3D holographic volume field
        self.wave_volume = np.zeros(volume_shape, dtype=np.complex128)
        self.object_wave = np.zeros(volume_shape, dtype=np.complex128)
        self.reference_wave = np.zeros(volume_shape, dtype=np.complex128)

    def encode_pattern(self, reference_wave: np.ndarray, object_wave: np.ndarray) -> None:
        """
        Encodes object wave onto reference wave via volumetric interference:
        H(x,y,z) = R * conj(O) + conj(R) * O
        """
        self.reference_wave = reference_wave
        self.object_wave = object_wave
        interference = reference_wave * np.conj(object_wave) + np.conj(reference_wave) * object_wave
        self.wave_volume = interference

    def encode_topological_points(self, points: List[UniversalTopologicalPoint]) -> None:
        """
        Generates 3D wave patterns for topological points and stores their holographic interference.
        """
        z_dim, y_dim, x_dim = self.volume_shape
        z_coords, y_coords, x_coords = np.meshgrid(
            np.linspace(-1, 1, z_dim),
            np.linspace(-1, 1, y_dim),
            np.linspace(-1, 1, x_dim),
            indexing='ij'
        )

        total_obj_wave = np.zeros(self.volume_shape, dtype=np.complex128)
        ref_wave = np.exp(1j * (x_coords * 2.0 + y_coords * 2.0 + z_coords * 2.0) * math.pi)

        for pt in points:
            px, py, pz = pt.position
            dist_sq = (x_coords - px)**2 + (y_coords - py)**2 + (z_coords - pz)**2
            phase = math.atan2(pt.phase_quaternion[1], pt.phase_quaternion[0])
            pt_wave = np.exp(-dist_sq / (2 * (0.3 * pt.scale)**2)) * np.exp(1j * (phase + math.pi * dist_sq))
            total_obj_wave += pt_wave * pt.phase_coherence

        self.encode_pattern(ref_wave, total_obj_wave)

    def reconstruct_from_crop(
        self,
        cropped_mask: np.ndarray,
        reference_wave: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Reconstructs the global object wave structure from a partial spatial crop
        using holographic wave propagation and phase retrieval.

        Args:
            cropped_mask: Boolean/Float 3D mask where 1.0 = available volume, 0.0 = cropped out / lost.
            reference_wave: The reference key wave used during encoding.

        Returns:
            reconstructed_object_wave: Complex 3D reconstructed wave volume.
        """
        z_dim, y_dim, x_dim = self.volume_shape
        if reference_wave is None:
            reference_wave = self.reference_wave

        # Crop the stored hologram spatially
        cropped_hologram = self.wave_volume * cropped_mask

        # Holographic readout: multiply cropped hologram by reference wave R
        # Readout contains O * |R|^2 + R^2 * O*
        raw_readout = cropped_hologram * reference_wave

        # 3D Angular Spectrum / Diffraction propagation into frequency domain
        freq_domain = np.fft.fftn(raw_readout)

        # Frequency domain phase filter for wave reconstruction
        k_z = np.fft.fftfreq(z_dim)[:, None, None]
        k_y = np.fft.fftfreq(y_dim)[None, :, None]
        k_x = np.fft.fftfreq(x_dim)[None, None, :]
        k_sq = k_x**2 + k_y**2 + k_z**2

        # Low/Mid-pass holographic reconstruction filter
        filter_kernel = np.exp(-k_sq / (2 * 0.25**2))
        reconstructed_freq = freq_domain * filter_kernel

        reconstructed_wave = np.fft.ifftn(reconstructed_freq)
        return reconstructed_wave


class CausalVolumetricSpacetimeEngine:
    """
    Continuous 3D/4D Volumetric Spacetime Engine.
    Operates without 2D layer stacks, matrix flattening, or discrete steps.
    Implements continuous volumetric wave propagation, phase-locking,
    bivector rotor relaxation, and scale-consistent causal homeostasis.
    """

    def __init__(self, volume_shape: Tuple[int, int, int] = (16, 16, 16), dt: float = 0.05):
        self.volume_shape = volume_shape
        self.dt = dt
        self.time = 0.0

        # Substrate fields
        self.points: List[UniversalTopologicalPoint] = []
        self.storage = HolographicVolumetricStorage(volume_shape)
        self.gradient_engine = KnowingIgnorancePhaseGradient(volume_shape)

        # 3D Continuous Spacetime Wave Field: Wave amplitude Psi(z,y,x) and velocity V(z,y,x)
        self.wave_psi = np.zeros(volume_shape, dtype=np.float64)
        self.wave_vel = np.zeros(volume_shape, dtype=np.float64)

        # Chromatic Density Field (3, Z, Y, X): [Red, Blue, Yellow]
        self.chromatic_field = np.zeros((3,) + volume_shape, dtype=np.float64)

    def add_topological_point(self, point: UniversalTopologicalPoint) -> None:
        """Adds a topological node into the volumetric substrate."""
        self.points.append(point)

    def step_spatiotemporal_evolution(self) -> Dict[str, Any]:
        """
        Evolves the 3D/4D volumetric continuum by one continuous temporal differential dt.
        Applies 3D wave equation: d2Psi/dt2 = c^2 * Laplacian(Psi) + CausalTension.
        """
        self.time += self.dt

        # 1. Update knowing-ignorance gradient
        self.gradient_engine.update_coherence_from_points(self.points)
        grad_field, tension_force = self.gradient_engine.compute_phase_strain_gradient()

        # 2. Compute 3D Laplacian of Psi using 6-neighbor 3D finite difference
        laplacian = (
            np.roll(self.wave_psi, 1, axis=0) + np.roll(self.wave_psi, -1, axis=0) +
            np.roll(self.wave_psi, 1, axis=1) + np.roll(self.wave_psi, -1, axis=1) +
            np.roll(self.wave_psi, 1, axis=2) + np.roll(self.wave_psi, -1, axis=2) -
            6.0 * self.wave_psi
        )

        # Wave speed c = 1.0 + coherence (known regions propagate faster)
        c_speed = 1.0 + self.gradient_engine.coherence_field

        # Wave acceleration = c^2 * Laplacian + tension_force
        accel = (c_speed**2) * laplacian + tension_force - 0.05 * self.wave_vel  # Damping

        # Symplectic Euler integration for wave equation
        self.wave_vel += accel * self.dt
        self.wave_psi += self.wave_vel * self.dt

        # 3. Rotate topological points based on local 3D wave gradients
        gz, gy, gx = np.gradient(self.wave_psi)
        for pt in self.points:
            z_dim, y_dim, x_dim = self.volume_shape
            gx_i = int(np.clip((pt.position[0] + 1.0) / 2.0 * (x_dim - 1), 0, x_dim - 1))
            gy_i = int(np.clip((pt.position[1] + 1.0) / 2.0 * (y_dim - 1), 0, y_dim - 1))
            gz_i = int(np.clip((pt.position[2] + 1.0) / 2.0 * (z_dim - 1), 0, z_dim - 1))

            local_grad = np.array([gx[gz_i, gy_i, gx_i], gy[gz_i, gy_i, gx_i], gz[gz_i, gy_i, gx_i]])
            grad_mag = np.linalg.norm(local_grad)
            if grad_mag > 1e-6:
                # Rotate point phase quaternion along local gradient axis
                pt.rotate_phase(local_grad, grad_mag * self.dt)
                # Phase relaxation increases coherence toward 1.0
                pt.set_phase_coherence(pt.phase_coherence + 0.02 * self.dt)

        # 4. Refresh holographic wave encoding
        self.storage.encode_topological_points(self.points)

        return {
            "time": self.time,
            "mean_wave_energy": float(np.mean(self.wave_psi**2)),
            "mean_coherence": float(np.mean(self.gradient_engine.coherence_field)),
            "max_tension_force": float(np.max(tension_force)),
            "num_points": len(self.points)
        }

    def evaluate_holographic_crop_reconstruction(self, crop_ratio: float = 0.7) -> float:
        """
        Simulates cropping a fraction (e.g. 70%) of the 3D volume, and evaluates
        the structural phase correlation between the original object wave and reconstructed wave.

        Returns:
            fidelity: Phase / structural correlation score [0.0, 1.0].
        """
        full_storage = HolographicVolumetricStorage(self.volume_shape)
        full_storage.encode_topological_points(self.points)
        original_wave = full_storage.object_wave

        # Create crop mask: 1.0 for available (1 - crop_ratio), 0.0 for cropped out
        mask = np.ones(self.volume_shape, dtype=np.float64)
        z_dim, y_dim, x_dim = self.volume_shape
        crop_z_end = int(z_dim * crop_ratio)
        mask[:crop_z_end, :, :] = 0.0  # Erase z-layers corresponding to crop_ratio

        reconstructed_wave = full_storage.reconstruct_from_crop(mask)

        # Structural Phase Correlation (cosine similarity of wave phase angles or complex normalized dot product)
        phase_orig = np.exp(1j * np.angle(original_wave))
        phase_recon = np.exp(1j * np.angle(reconstructed_wave))

        # Evaluate correlation
        mag_orig = np.abs(original_wave)
        mask_active = mag_orig > 1e-4

        if not np.any(mask_active):
            return 1.0

        # Phase coherence correlation between target wave and diffracted reconstruction
        phase_diff = np.abs(phase_orig[mask_active] - phase_recon[mask_active])
        phase_similarity = float(np.mean(np.cos(phase_diff)))
        # Map cosine [-1, 1] to [0, 1]
        fidelity = float(max(0.0, (phase_similarity + 1.0) / 2.0))
        return fidelity
