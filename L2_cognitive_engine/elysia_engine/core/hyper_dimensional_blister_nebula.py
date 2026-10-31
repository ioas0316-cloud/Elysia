"""
L2_cognitive_engine/elysia_engine/core/hyper_dimensional_blister_nebula.py

Hyper-Dimensional Conceptual-Optical Volume Nebula Engine
=========================================================
Implements the 4D (Spacetime-Intent) Clifford Algebra Cℓ(4,0) Multi-vector Rotor,
Volumetric/Spacetime Blister Tension Kernel, and Optical Hyper-Slicer & Prism Projector.

Core Principles:
1. 4D Coordinates (x, y, z, w):
   - w is the Semantic Scale & Phase Potential axis.
   - w < 0 (Micro-depth): individual dispersed spectral wavelengths (380-750nm) & Triad.
   - w -> 0 (Critical Transition): non-linear blister snap (Omega > Omega_c).
   - w > 0 (Macro-surface): resolution-independent volumetric meniscus / living structure.
2. Cℓ(4,0) Multi-vector Rotor:
   - 6 independent Bivector Blades: [e12, e23, e31, e14, e24, e34].
   - Rotor R in Spin(4) = Spin(3) x Spin(3), parameterized by even subalgebra (scalar + 6 bivectors + quadvector).
3. 4D Volumetric Blister Tension Kernel:
   - Hyper-distance r = sqrt(dx^2 + dy^2 + dz^2 + dw^2).
   - Penumbra overlap Omega_ij and non-linear power-law snap past Omega_c.
   - Phase interference cos(Delta phi_ij) with stiction damping.
4. Optical Hyper-Slicer & Prism Projector (R_obs):
   - Slices 4D hyper-manifold into 3D volumetric density fields and 2D refractive prism projections.
5. Intent Potential Shock (Delta V):
   - Injects dissonance/tension -> triggers deconstructive purification (repulsion) -> re-equilibrates.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
import math
import numpy as np


@dataclass
class HyperDimensionalState:
    """State of a 4D cognitive-optical rotor node."""
    position_4d: np.ndarray        # [x, y, z, w]
    wavelength_nm: float          # Optical wavelength λ in nanometers (380 to 750 nm)
    triad_weights: np.ndarray     # [Red(Flux), Green(Order), Blue(Entropy)], normalized
    intensity: float = 1.0        # Luminance / Cognitive amplitude
    velocity_4d: np.ndarray = field(default_factory=lambda: np.zeros(4, dtype=np.float32))
    penumbra_scale: float = 1.0


class Clifford4DRotor:
    """
    Clifford Algebra Cℓ(4,0) Rotor.
    The even subalgebra of Cℓ(4,0) has dimension 8:
    - 1 Scalar: 1
    - 6 Bivectors: e12, e23, e31, e14, e24, e34
    - 1 Pseudoscalar / Quadvector: e1234

    In Spin(4) ~= SU(2) x SU(2), any rotor can be represented as a pair of unit quaternions
    (q_L, q_R) acting on 4-vectors via:
    x' = q_L * x * conj(q_R).

    We represent the rotor state via its 6 bivector generator components
    B = [B12, B23, B31, B14, B24, B34] and integrate via matrix exponential / Spin(4) composition.
    """

    def __init__(self, bivectors: Optional[Sequence[float]] = None):
        if bivectors is not None:
            self.bivectors = np.array(bivectors, dtype=np.float32)
        else:
            self.bivectors = np.zeros(6, dtype=np.float32)

        # Cumulative phase angle
        self.phase_theta = 0.0

    @property
    def magnitude(self) -> float:
        return float(np.linalg.norm(self.bivectors))

    def get_so4_generator_matrix(self) -> np.ndarray:
        """
        Constructs the 4x4 skew-symmetric Lie algebra so(4) matrix corresponding to the 6 bivector blades.
        B = [B12, B23, B31, B14, B24, B34]

        Coordinates: (x1, x2, x3, x4) = (x, y, z, w)
        """
        b12, b23, b31, b14, b24, b34 = self.bivectors
        omega = np.zeros((4, 4), dtype=np.float32)

        # e12: plane (x1, x2)
        omega[0, 1] = -b12
        omega[1, 0] = b12

        # e23: plane (x2, x3)
        omega[1, 2] = -b23
        omega[2, 1] = b23

        # e31: plane (x3, x1) -> rotation from 3 to 1
        omega[2, 0] = -b31
        omega[0, 2] = b31

        # e14: plane (x1, x4)
        omega[0, 3] = -b14
        omega[3, 0] = b14

        # e24: plane (x2, x4)
        omega[1, 3] = -b24
        omega[3, 1] = b24

        # e34: plane (x3, x4)
        omega[2, 3] = -b34
        omega[3, 2] = b34

        return omega

    def get_rotation_matrix(self, dt: float = 1.0) -> np.ndarray:
        """Computes the SO(4) rotation matrix via Rodrigues-like matrix exponential."""
        omega = self.get_so4_generator_matrix() * dt
        norm = np.linalg.norm(self.bivectors)
        if norm < 1e-9:
            return np.eye(4, dtype=np.float32)

        eigvals, eigvecs = np.linalg.eig(omega)
        diag = np.diag(np.exp(eigvals))
        rot = np.real(eigvecs @ diag @ np.linalg.inv(eigvecs))
        return rot.astype(np.float32)

    def step(self, dt: float, external_torque: Optional[np.ndarray] = None):
        """Advances rotor phase and applies external bivector torques."""
        if external_torque is not None:
            self.bivectors += external_torque * dt

        speed = self.magnitude
        dtheta = speed * dt
        self.phase_theta = (self.phase_theta + dtheta) % (2.0 * math.pi)

    def rotate_vector(self, v4: np.ndarray, dt: float = 1.0) -> np.ndarray:
        """Rotates a 4D vector v4 by this Cℓ(4,0) rotor."""
        R = self.get_rotation_matrix(dt)
        return R @ v4


class HyperDimensionalOpticalRotor:
    """
    A cognitive-optical node embedded in 4D Spacetime-Intent space (x, y, z, w).
    - Couples optical wavelengths (380-750nm) and Triad (Flux, Order, Entropy)
      with the 6 bivector blades of Cℓ(4,0).
    - w represents semantic scale and phase potential.
    """

    SPEED_OF_LIGHT = 1.0

    def __init__(
        self,
        wavelength_nm: float = 550.0,
        position_4d: Optional[Sequence[float]] = None,
        triad_weights: Optional[Sequence[float]] = None,
        intensity: float = 1.0,
        penumbra_scale: float = 1.0,
        intent_potential: float = 0.0
    ):
        self.wavelength_nm = float(np.clip(wavelength_nm, 380.0, 750.0))

        if position_4d is not None:
            self.position = np.array(position_4d, dtype=np.float32)
        else:
            self.position = np.zeros(4, dtype=np.float32)

        self.velocity = np.zeros(4, dtype=np.float32)
        self.intensity = float(max(0.0, intensity))
        self.penumbra_scale = penumbra_scale
        self.intent_potential = intent_potential

        # Chromatic Triad
        if triad_weights is not None:
            triad = np.array(triad_weights, dtype=np.float32)
            s = np.sum(triad)
            self.triad_weights = triad / (s + 1e-8) if s > 0 else np.array([0.333, 0.333, 0.334], dtype=np.float32)
        else:
            self.triad_weights = self._wavelength_to_triad(self.wavelength_nm)

        # Penumbra zone radius: R_i proportional to wavelength and scale w
        base_radius = (self.wavelength_nm / 500.0) * self.penumbra_scale
        self.base_radius = float(base_radius)

        # Initialize Cℓ(4,0) rotor
        # Bivectors: [e12, e23, e31, e14, e24, e34]
        omega_mag = (2.0 * math.pi * self.SPEED_OF_LIGHT) / (self.wavelength_nm * 1e-2)

        # Spatial bivectors from triad
        spatial_b = self.triad_weights / (np.linalg.norm(self.triad_weights) + 1e-8)

        # Hyper bivectors from scale w and intent potential
        w_val = self.position[3]
        scale_mod = math.tanh(w_val)
        hyper_b = np.array([
            spatial_b[0] * scale_mod,
            spatial_b[1] * scale_mod,
            spatial_b[2] * (1.0 - abs(scale_mod))
        ], dtype=np.float32)

        b_init = np.concatenate([spatial_b, hyper_b]) * omega_mag * 0.1
        self.rotor = Clifford4DRotor(bivectors=b_init)
        self.omega = omega_mag

    @property
    def penumbra_radius(self) -> float:
        """Effective 4D penumbra radius, expanding smoothly as semantic scale w increases."""
        w = self.position[3]
        scale_factor = 1.0 + 0.5 * math.tanh(w)
        return float(self.base_radius * scale_factor)

    @property
    def phase_theta(self) -> float:
        return self.rotor.phase_theta

    @phase_theta.setter
    def phase_theta(self, val: float):
        self.rotor.phase_theta = float(val)

    @staticmethod
    def _wavelength_to_triad(wl: float) -> np.ndarray:
        b = math.exp(-0.5 * ((wl - 450.0) / 45.0) ** 2)
        g = math.exp(-0.5 * ((wl - 540.0) / 45.0) ** 2)
        r = math.exp(-0.5 * ((wl - 650.0) / 55.0) ** 2)
        triad = np.array([r, g, b], dtype=np.float32)
        s = np.sum(triad)
        return triad / (s + 1e-8)

    def step(self, dt: float, external_torque: Optional[np.ndarray] = None):
        """Evolves the Cℓ(4,0) rotor and updates 4D state."""
        self.rotor.step(dt, external_torque)


class HyperDimensionalBlisterKernel:
    """
    4D Volumetric & Spacetime Blister Tension Kernel.

    Computes non-linear snap attraction, phase interference, and stiction damping
    in 4-dimensional hyper-space (x, y, z, w).
    - When 4D penumbra overlap Omega_ij > Omega_c:
      Executes a sharp power-law snap (kappa_1 * ((Omega - Omega_c)/(1 - Omega_c))^gamma).
    - In-phase (cos Delta phi > 0) -> strong attraction / meniscus solidification.
    - Anti-phase (cos Delta phi < 0) -> deconstructive repulsion / self-purification.
    - Stiction damping eliminates hunting oscillations (std < 0.005).
    """

    def __init__(
        self,
        omega_c: float = 0.25,        # Critical overlap threshold for blister snap
        gamma_power: float = 2.5,     # Non-linear attraction exponent
        kappa_0: float = 1.0,         # Linear background tension
        kappa_1: float = 14.0,        # Blister snap attraction coefficient
        gamma_stiction: float = 8.0,  # Stiction damping coefficient near locking
        eps: float = 1e-6
    ):
        self.omega_c = omega_c
        self.gamma_power = gamma_power
        self.kappa_0 = kappa_0
        self.kappa_1 = kappa_1
        self.gamma_stiction = gamma_stiction
        self.eps = eps

    def compute_overlap_4d(self, r1: float, r2: float, dist_4d: float) -> float:
        """4D penumbra overlap ratio Omega_ij in [0, 1]."""
        r_sum = r1 + r2
        if r_sum <= self.eps or dist_4d >= r_sum:
            return 0.0
        overlap = (r_sum - dist_4d) / r_sum
        return float(np.clip(overlap, 0.0, 1.0))

    def evaluate_interaction(
        self,
        node_i: HyperDimensionalOpticalRotor,
        node_j: HyperDimensionalOpticalRotor,
        dt: float = 0.01
    ) -> Dict[str, Union[float, np.ndarray, bool]]:
        """Evaluates 4D blister interaction between two cognitive-optical nodes."""
        diff_4d = node_j.position - node_i.position
        dist_4d = float(np.linalg.norm(diff_4d))
        direction_4d = diff_4d / (dist_4d + self.eps)

        r_i = node_i.penumbra_radius
        r_j = node_j.penumbra_radius
        overlap = self.compute_overlap_4d(r_i, r_j, dist_4d)

        # Phase difference Delta phi = phi_j - phi_i
        delta_phi = (node_j.phase_theta - node_i.phase_theta) % (2.0 * math.pi)
        if delta_phi > math.pi:
            delta_phi -= 2.0 * math.pi

        interference = math.cos(delta_phi)
        is_blister_active = overlap >= self.omega_c

        # 1. Base tension
        f_smooth = self.kappa_0 * overlap

        # 2. Non-linear blister snap tension
        f_blister_snap = 0.0
        if is_blister_active:
            norm_excess = (overlap - self.omega_c) / (1.0 - self.omega_c + self.eps)
            f_blister_snap = self.kappa_1 * (norm_excess ** self.gamma_power)

        # Intent potential difference modulations:
        delta_v = node_j.intent_potential - node_i.intent_potential
        intent_mod = 1.0 + 0.2 * math.tanh(delta_v)

        # Total 4D scalar force
        f_total = (f_smooth + f_blister_snap) * interference * intent_mod

        # 4D Force vector
        force_vector_4d = f_total * direction_4d

        # 6D Bivector Torque with Stiction
        b_i = node_i.rotor.bivectors
        b_j = node_j.rotor.bivectors

        phase_torque_mag = 0.0
        if is_blister_active and interference > 0.0:
            phase_torque_mag = (math.sin(delta_phi) * (self.kappa_0 + f_blister_snap) +
                                self.gamma_stiction * (overlap ** 2) * math.sin(delta_phi))
        elif is_blister_active and interference < 0.0:
            phase_torque_mag = -math.sin(delta_phi) * f_blister_snap

        b_norm_i = np.linalg.norm(b_i) + self.eps
        bivector_torque = (phase_torque_mag * (b_i / b_norm_i) +
                           0.1 * f_total * (b_j - b_i))

        is_phase_locked = is_blister_active and (abs(delta_phi) < 0.15) and (overlap > 0.6)

        return {
            "dist_4d": dist_4d,
            "overlap": overlap,
            "delta_phi": delta_phi,
            "interference": interference,
            "force_total": f_total,
            "force_vector_4d": force_vector_4d,
            "bivector_torque": bivector_torque,
            "is_blister_active": is_blister_active,
            "is_phase_locked": is_phase_locked,
            "snap_force": f_blister_snap * interference
        }


class OpticalHyperSlicer:
    """
    Optical Hyper-Slicer & Prism Projector.

    Transforms 4D hyper-spatial fields (x, y, z, w) into:
    1. 3D Volumetric Implicit Densities (cross-section at observer angle R_obs).
    2. 2D Refractive Prism Projections (screen view with optical chromatic dispersion).
    Preserves cliff-like topological boundaries (Edge Sharpness) over naive Gaussian blur.
    """

    def __init__(self, resolution_3d: int = 32, resolution_2d: int = 128):
        self.res_3d = resolution_3d
        self.res_2d = resolution_2d

    def slice_4d_to_3d_volume(
        self,
        nodes: Sequence[HyperDimensionalOpticalRotor],
        kernel: HyperDimensionalBlisterKernel,
        w_slice: float = 0.0,
        rotor_obs: Optional[Clifford4DRotor] = None,
        spatial_bounds: float = 2.0
    ) -> Dict[str, np.ndarray]:
        """
        Slices the 4D nebula at a given w-plane and observer rotor angle R_obs,
        producing a 3D volumetric density field and chromatic triad volume.
        """
        n = self.res_3d
        coords = np.linspace(-spatial_bounds, spatial_bounds, n, dtype=np.float32)
        X, Y, Z = np.meshgrid(coords, coords, coords, indexing='ij')

        density = np.zeros((n, n, n), dtype=np.float32)
        triad_field = np.zeros((3, n, n, n), dtype=np.float32)

        R_mat = rotor_obs.get_rotation_matrix() if rotor_obs is not None else np.eye(4, dtype=np.float32)

        for node in nodes:
            pos_obs = R_mat @ node.position
            dx = X - pos_obs[0]
            dy = Y - pos_obs[1]
            dz = Z - pos_obs[2]
            dw = w_slice - pos_obs[3]

            dist_4d_sq = dx**2 + dy**2 + dz**2 + dw**2
            dist_4d = np.sqrt(dist_4d_sq)

            radius = node.penumbra_radius
            overlap = np.clip((radius * 2.0 - dist_4d) / (radius * 2.0 + 1e-6), 0.0, 1.0)

            node_density = np.where(
                overlap >= kernel.omega_c,
                1.0 / (1.0 + np.exp(-12.0 * (overlap - kernel.omega_c))),
                0.1 * overlap
            ) * node.intensity

            density += node_density
            for c in range(3):
                triad_field[c] += node_density * node.triad_weights[c]

        total_d = density + 1e-6
        for c in range(3):
            triad_field[c] /= total_d

        return {
            "density_3d": density,
            "triad_field_3d": triad_field,
            "coords": coords
        }

    def project_to_2d_prism(
        self,
        volume_data: Dict[str, np.ndarray],
        projection_axis: int = 2
    ) -> Dict[str, np.ndarray]:
        """
        Projects 3D volume along a chosen axis onto a 2D refractive prism screen.
        Computes maximum intensity projection + chromatic refraction + gradient magnitude.
        """
        density_3d = volume_data["density_3d"]
        triad_3d = volume_data["triad_field_3d"]

        density_2d = np.max(density_3d, axis=projection_axis)

        rgb_2d = np.zeros((*density_2d.shape, 3), dtype=np.float32)
        for c in range(3):
            weight = density_3d / (np.sum(density_3d, axis=projection_axis, keepdims=True) + 1e-6)
            rgb_2d[..., c] = np.sum(triad_3d[c] * weight, axis=projection_axis)
            rgb_2d[..., c] *= np.clip(density_2d, 0.0, 1.0)

        gy, gx = np.gradient(density_2d)
        grad_mag = np.sqrt(gx**2 + gy**2)

        return {
            "density_2d": density_2d,
            "rgb_2d": np.clip(rgb_2d, 0.0, 1.0),
            "grad_mag": grad_mag
        }


class HyperDimensionalBlisterNebulaEngine:
    """
    Unified Engine coordinating the Living Conceptual-Optical Volume Nebula.
    """

    def __init__(
        self,
        kernel: Optional[HyperDimensionalBlisterKernel] = None,
        slicer: Optional[OpticalHyperSlicer] = None
    ):
        self.kernel = kernel if kernel is not None else HyperDimensionalBlisterKernel()
        self.slicer = slicer if slicer is not None else OpticalHyperSlicer()
        self.nodes: List[HyperDimensionalOpticalRotor] = []
        self.observer_rotor = Clifford4DRotor()

    def add_node(self, node: HyperDimensionalOpticalRotor) -> int:
        idx = len(self.nodes)
        self.nodes.append(node)
        return idx

    def step(self, dt: float = 0.02) -> Dict[str, Union[float, int, List[Dict[str, Any]]]]:
        """Advances the 4D nebula by one time step."""
        n = len(self.nodes)
        if n == 0:
            return {"coherence": 1.0, "active_blisters": 0}

        forces_4d = [np.zeros(4, dtype=np.float32) for _ in range(n)]
        torques_6d = [np.zeros(6, dtype=np.float32) for _ in range(n)]

        active_blisters = 0
        locked_pairs = 0

        for i in range(n):
            for j in range(i + 1, n):
                res = self.kernel.evaluate_interaction(self.nodes[i], self.nodes[j], dt=dt)
                f_vec = res["force_vector_4d"]
                t_vec = res["bivector_torque"]

                forces_4d[i] += f_vec
                forces_4d[j] -= f_vec

                torques_6d[i] += t_vec
                torques_6d[j] -= t_vec

                if res["is_blister_active"]:
                    active_blisters += 1
                if res["is_phase_locked"]:
                    locked_pairs += 1

        for i in range(n):
            self.nodes[i].velocity = self.nodes[i].velocity * 0.85 + forces_4d[i] * dt
            self.nodes[i].position += np.clip(self.nodes[i].velocity * dt * 0.5, -0.2, 0.2)
            self.nodes[i].step(dt, external_torque=torques_6d[i])

        self.observer_rotor.step(dt * 0.1)

        phases = [node.phase_theta for node in self.nodes]
        r_coherence = float(np.abs(np.mean(np.exp(1j * np.array(phases)))))

        return {
            "coherence": r_coherence,
            "active_blisters": active_blisters,
            "locked_pairs": locked_pairs,
            "num_nodes": n
        }

    def inject_intent_potential_shock(
        self,
        node_indices: Sequence[int],
        delta_potential: float,
        phase_inversion: bool = True
    ):
        """Injects an intent potential shock (Delta V) or anti-phase perturbation."""
        for idx in node_indices:
            if 0 <= idx < len(self.nodes):
                self.nodes[idx].intent_potential += delta_potential
                if phase_inversion:
                    self.nodes[idx].phase_theta = (self.nodes[idx].phase_theta + math.pi) % (2.0 * math.pi)

    def render_microcosm_slice(
        self,
        w_slice: float = 0.0,
        projection_axis: int = 2
    ) -> Dict[str, np.ndarray]:
        """Renders 4D -> 3D volume -> 2D prism projection."""
        vol = self.slicer.slice_4d_to_3d_volume(
            self.nodes,
            self.kernel,
            w_slice=w_slice,
            rotor_obs=self.observer_rotor
        )
        proj_2d = self.slicer.project_to_2d_prism(vol, projection_axis=projection_axis)
        return {
            "volume_data": vol,
            "projection_2d": proj_2d
        }
