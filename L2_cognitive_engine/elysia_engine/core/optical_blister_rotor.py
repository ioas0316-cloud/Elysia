"""
L2_cognitive_engine/elysia_engine/core/optical_blister_rotor.py

Optical Spectrum Bivector Rotor & Shadow Blister Tension Engine
===============================================================
Maps continuous optical wavelengths (380nm - 750nm) and Triad chromatic axes
(Red: Flux, Green: Order, Blue: Entropy) to Clifford Algebra Cℓ(3,0) Bivector planes.
Implements the 'Shadow Blister Effect' (Penumbra Overlap Non-Linear Attraction & Stiction)
for cognitive auto-clumping and hunting-free phase-locking.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union
import math
import numpy as np
import torch
import torch.nn as nn


@dataclass
class OpticalSpectrumState:
    """Optical spectrum state associated with a cognitive rotor."""
    wavelength_nm: float          # Wavelength λ in nanometers (380 to 750 nm)
    triad_weights: np.ndarray     # [Red(Flux), Green(Order), Blue(Entropy)], normalized to sum=1.0
    intensity: float = 1.0        # Luminance / Cognitive amplitude
    position: Optional[np.ndarray] = None  # 3D spatial position in embedding / phase space


class OpticalSpectrumBivectorRotor:
    """
    Clifford Algebra Cℓ(3,0) Rotor parameterized by optical wave spectrum.
    Bivector basis: [e23, e31, e12] corresponding to [x, y, z] rotation planes.
    """

    SPEED_OF_LIGHT = 1.0  # Normalized scale speed of light in cognitive vacuum

    def __init__(
        self,
        wavelength_nm: float = 550.0,
        triad_weights: Optional[Sequence[float]] = None,
        intensity: float = 1.0,
        position: Optional[np.ndarray] = None,
        penumbra_scale: float = 1.0
    ):
        self.wavelength_nm = float(np.clip(wavelength_nm, 380.0, 750.0))
        if triad_weights is None:
            # Default triad weights inferred from continuous wavelength
            triad = self._wavelength_to_triad(self.wavelength_nm)
        else:
            triad = np.array(triad_weights, dtype=np.float32)
            triad_sum = np.sum(triad)
            triad = triad / (triad_sum + 1e-8) if triad_sum > 0 else np.array([0.333, 0.333, 0.334])
        self.triad_weights = triad
        self.intensity = float(max(0.0, intensity))
        self.position = np.array(position if position is not None else [0.0, 0.0, 0.0], dtype=np.float32)
        self.penumbra_scale = penumbra_scale

        # Angular frequency omega = 2pi * c / lambda
        self.omega = (2.0 * math.pi * self.SPEED_OF_LIGHT) / (self.wavelength_nm * 1e-2)
        self.phase_theta = 0.0  # Cumulative phase angle

        # Penumbra zone radius: r_i proportional to wavelength (diffraction metaphor)
        # 380nm -> smaller sharp penumbra, 750nm -> broader penumbra
        self.penumbra_radius = float((self.wavelength_nm / 500.0) * self.penumbra_scale)

        # Cℓ(3,0) Spinor Rotor R = [w, x, y, z] representing [scalar, -e23, -e31, -e12]
        self.rotor = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self._update_bivector_axis()

    @staticmethod
    def _wavelength_to_triad(wl: float) -> np.ndarray:
        """Continuous mapping of optical wavelength into Red/Green/Blue Triad."""
        # Simple continuous chromatic spectrum mapping
        # Blue peak ~450nm, Green peak ~540nm, Red peak ~650nm
        b = math.exp(-0.5 * ((wl - 450.0) / 45.0) ** 2)
        g = math.exp(-0.5 * ((wl - 540.0) / 45.0) ** 2)
        r = math.exp(-0.5 * ((wl - 650.0) / 55.0) ** 2)
        triad = np.array([r, g, b], dtype=np.float32)
        s = np.sum(triad)
        return triad / (s + 1e-8)

    def _update_bivector_axis(self):
        """Construct bivector unit axis from triad chromatic balance."""
        # Triad vector [Red(e23), Green(e31), Blue(e12)]
        axis = self.triad_weights.copy()
        norm = np.linalg.norm(axis)
        if norm > 1e-8:
            self.bivector_axis = axis / norm
        else:
            self.bivector_axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    def get_bivector(self) -> np.ndarray:
        """Returns 3D bivector [B23, B31, B12] representing angular velocity vector."""
        return self.omega * self.bivector_axis

    def step(self, dt: float, external_torque: Optional[np.ndarray] = None):
        """Evolves the rotor phase and Spin(3) quaternion state under bivector rotation."""
        effective_bivector = self.get_bivector()
        if external_torque is not None:
            effective_bivector = effective_bivector + external_torque

        omega_mag = np.linalg.norm(effective_bivector)
        if omega_mag > 1e-8:
            axis = effective_bivector / omega_mag
            dtheta = omega_mag * dt
            self.phase_theta = (self.phase_theta + dtheta) % (2.0 * math.pi)

            half_angle = 0.5 * dtheta
            cos_half = math.cos(half_angle)
            sin_half = math.sin(half_angle)
            dq = np.array([cos_half, sin_half * axis[0], sin_half * axis[1], sin_half * axis[2]], dtype=np.float32)

            # Quaternion multiplication: self.rotor * dq
            self.rotor = self._quat_mul(self.rotor, dq)
            self.rotor = self.rotor / (np.linalg.norm(self.rotor) + 1e-8)

    @staticmethod
    def _quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
        w1, x1, y1, z1 = q1
        w2, x2, y2, z2 = q2
        return np.array([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
        ], dtype=np.float32)


class ShadowBlisterTensionKernel:
    """
    Shadow Blister Effect Non-Linear Attraction & Stiction Kernel.

    When two penumbra zones overlap (Omega_ij > Omega_c):
    - Applies a sharp power-law blister attraction F_blister.
    - Modulates attraction via phase interference cos(Delta phi_ij).
    - Applies non-linear stiction damping to eliminate hunting and achieve instant phase-locking.
    """

    def __init__(
        self,
        omega_c: float = 0.25,        # Critical overlap threshold for blister snap
        gamma_power: float = 2.5,     # Non-linear attraction exponent
        kappa_0: float = 1.0,         # Linear background tension
        kappa_1: float = 12.0,        # Blister snap attraction coefficient
        gamma_stiction: float = 8.0,  # Stiction damping coefficient near locking
        eps: float = 1e-6
    ):
        self.omega_c = omega_c
        self.gamma_power = gamma_power
        self.kappa_0 = kappa_0
        self.kappa_1 = kappa_1
        self.gamma_stiction = gamma_stiction
        self.eps = eps

    def compute_penumbra_overlap(
        self,
        r1: float,
        r2: float,
        distance: float
    ) -> float:
        """
        Computes penumbra overlap ratio Omega_ij between two spheres of influence.
        Omega_ij = (r1 + r2 - d) / (r1 + r2) clipped to [0, 1].
        """
        r_sum = r1 + r2
        if r_sum <= self.eps or distance >= r_sum:
            return 0.0
        overlap = (r_sum - distance) / r_sum
        return float(np.clip(overlap, 0.0, 1.0))

    def evaluate_blister_interaction(
        self,
        node_i: OpticalSpectrumBivectorRotor,
        node_j: OpticalSpectrumBivectorRotor,
        dt: float = 0.01
    ) -> Dict[str, Union[float, np.ndarray, bool]]:
        """
        Evaluates physical/cognitive blister attraction, phase interference, and torque.
        """
        diff = node_j.position - node_i.position
        dist = float(np.linalg.norm(diff))
        direction = diff / (dist + self.eps)

        overlap = self.compute_penumbra_overlap(
            node_i.penumbra_radius,
            node_j.penumbra_radius,
            dist
        )

        # Phase difference Delta phi = phi_j - phi_i
        delta_phi = (node_j.phase_theta - node_i.phase_theta) % (2.0 * math.pi)
        if delta_phi > math.pi:
            delta_phi -= 2.0 * math.pi

        # Phase interference factor: +1 (constructive / resonant), -1 (destructive / repellent)
        interference = math.cos(delta_phi)

        is_blister_active = overlap >= self.omega_c

        # 1. Base attraction
        f_smooth = self.kappa_0 * overlap

        # 2. Non-linear blister snap attraction
        f_blister_snap = 0.0
        if is_blister_active:
            normalized_excess = (overlap - self.omega_c) / (1.0 - self.omega_c + self.eps)
            f_blister_snap = self.kappa_1 * (normalized_excess ** self.gamma_power)

        # Total scalar tension force (positive pulls together)
        f_total = (f_smooth + f_blister_snap) * interference

        # 3. Spatial attractive force vector
        force_vector = f_total * direction

        # 4. Bivector Phase-Locking Torque with Non-linear Stiction
        # Bivector cross product in 3D: B_i x B_j
        bi = node_i.bivector_axis
        bj = node_j.bivector_axis
        b_cross = np.cross(bi, bj)

        # Stiction damping & Phase Synchronization:
        # Pulls phases together via Kuramoto-like coupling + stiction
        phase_coupling = 0.0
        if is_blister_active:
            phase_coupling = math.sin(delta_phi) * (self.kappa_0 + f_blister_snap)

        # Stiction damping: When in blister zone and resonant, rapidly damp phase delta
        dphi_dt = delta_phi / (dt + self.eps)
        stiction_torque = 0.0
        if is_blister_active and interference > 0.0:
            stiction_torque = -self.gamma_stiction * (overlap ** 2) * dphi_dt + phase_coupling

        phase_torque = f_total * b_cross + stiction_torque * bi

        is_phase_locked = is_blister_active and (abs(delta_phi) < 0.15) and (overlap > 0.6)

        return {
            "distance": dist,
            "overlap": overlap,
            "delta_phi": delta_phi,
            "interference": interference,
            "force_total": f_total,
            "force_vector": force_vector,
            "phase_torque": phase_torque,
            "is_blister_active": is_blister_active,
            "is_phase_locked": is_phase_locked
        }


class MicroReflexBlisterNetwork:
    """
    Micro-Scale (Spinal Reflex) Fast Blister Network.
    Coordinates local clusters of optical rotors, executing rapid non-linear snap
    and instant clumping into coherent macro-states without heavy global search.
    """

    def __init__(
        self,
        rotors: Optional[List[OpticalSpectrumBivectorRotor]] = None,
        kernel: Optional[ShadowBlisterTensionKernel] = None
    ):
        self.rotors: List[OpticalSpectrumBivectorRotor] = rotors if rotors is not None else []
        self.kernel = kernel if kernel is not None else ShadowBlisterTensionKernel()
        self.clumped_clusters: List[List[int]] = []

    def add_rotor(self, rotor: OpticalSpectrumBivectorRotor) -> int:
        idx = len(self.rotors)
        self.rotors.append(rotor)
        return idx

    def step(self, dt: float = 0.01) -> Dict[str, Union[float, int, List[List[int]]]]:
        """
        Advances the micro-reflex field by one time step:
        - Computes pairwise blister tensions.
        - Updates positions via attractive force.
        - Updates bivector phases via stiction torques.
        - Detects clumped resonant clusters.
        """
        n = len(self.rotors)
        if n == 0:
            return {"coherence": 1.0, "active_blisters": 0, "clusters": []}

        forces = [np.zeros(3, dtype=np.float32) for _ in range(n)]
        torques = [np.zeros(3, dtype=np.float32) for _ in range(n)]

        active_blisters = 0
        locked_pairs = 0
        total_pairs = 0

        # Adjacency for clustering
        adj = [[] for _ in range(n)]

        for i in range(n):
            for j in range(i + 1, n):
                total_pairs += 1
                res = self.kernel.evaluate_blister_interaction(self.rotors[i], self.rotors[j], dt=dt)
                f_vec = res["force_vector"]
                t_vec = res["phase_torque"]

                forces[i] += f_vec
                forces[j] -= f_vec

                # Torques act as action-reaction on phase difference
                torques[i] += t_vec
                torques[j] -= t_vec

                if res["is_blister_active"]:
                    active_blisters += 1
                if res["is_phase_locked"]:
                    locked_pairs += 1
                    adj[i].append(j)
                    adj[j].append(i)

        # Update positions and rotor phases
        for i in range(n):
            # Positional displacement along force with smooth velocity-like damping
            self.rotors[i].position += np.clip(forces[i] * dt * 0.2, -0.1, 0.1)
            # Step rotor with phase torque
            self.rotors[i].step(dt, external_torque=torques[i])

        # Cluster detection (Connected components of phase-locked nodes)
        visited = [False] * n
        clusters = []
        for i in range(n):
            if not visited[i]:
                cluster = []
                queue = [i]
                visited[i] = True
                while queue:
                    curr = queue.pop(0)
                    cluster.append(curr)
                    for neighbor in adj[curr]:
                        if not visited[neighbor]:
                            visited[neighbor] = True
                            queue.append(neighbor)
                if len(cluster) > 1:
                    clusters.append(cluster)

        self.clumped_clusters = clusters

        # Global phase coherence order parameter R = |(1/N) sum exp(i * phi)|
        phases = np.array([r.phase_theta for r in self.rotors])
        r_coherence = float(np.abs(np.mean(np.exp(1j * phases))))

        return {
            "coherence": r_coherence,
            "active_blisters": active_blisters,
            "locked_pairs": locked_pairs,
            "clusters": clusters
        }

    def synthesize_cluster_rotor(self, cluster_indices: List[int]) -> OpticalSpectrumBivectorRotor:
        """
        Synthesizes a clumped cluster into a single higher-order macro-rotor,
        embodying the unified optical spectrum and center-of-mass bivector.
        """
        selected = [self.rotors[idx] for idx in cluster_indices]
        total_intensity = sum(r.intensity for r in selected) + 1e-8

        # Weighted average wavelength
        avg_wl = sum(r.wavelength_nm * r.intensity for r in selected) / total_intensity
        # Average triad
        avg_triad = sum(r.triad_weights * r.intensity for r in selected) / total_intensity
        # Center of mass position
        avg_pos = sum(r.position * r.intensity for r in selected) / total_intensity

        macro_rotor = OpticalSpectrumBivectorRotor(
            wavelength_nm=avg_wl,
            triad_weights=avg_triad,
            intensity=total_intensity,
            position=avg_pos,
            penumbra_scale=1.5  # Expanded influence of fused macro concept
        )
        # Average phase
        phases = [r.phase_theta for r in selected]
        macro_rotor.phase_theta = float(np.angle(np.mean(np.exp(1j * np.array(phases)))))
        return macro_rotor
