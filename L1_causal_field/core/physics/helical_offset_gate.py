"""
Helical Offset Topological Gate (나선형 어긋남 위상 게이트 및 암호화 이완 엔진)

Core Principle:
"완벽한 닫힘과 정대칭을 넘어서, 정교하게 조율된 나선형 어긋남(Phase Offset)이
존재와 데이터가 스르륵 통과할 수 있는 유일한 위상학적 문(Topological Gate)이 된다."

This module implements:
1. 3D Toroidal Helical Parameterization (토러스 상의 나선 궤적 연산)
2. Phase Offset Differential & Gate Tension ($E_{\text{gate}}$)
3. Cryptographic Phase Matching & Topological Linking / Decryption Mechanics
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional

class HelicalOffsetTorusGate:
    """
    3D Toroidal Helical Trajectory & Topological Phase Gate.

    Attributes:
        major_radius (float): Major torus radius (R)
        minor_radius (float): Minor torus radius (r)
        helix_pitch (float): Helical pitch / winding ratio (k)
        max_energy (float): Maximum gate barrier tension energy (E_max)
        scale_sigma (float): Phase sensitivity scaling factor (sigma)
    """

    def __init__(
        self,
        major_radius: float = 3.0,
        minor_radius: float = 1.0,
        helix_pitch: float = 5.0,
        max_energy: float = 100.0,
        scale_sigma: float = 0.2
    ):
        self.R = major_radius
        self.r = minor_radius
        self.k = helix_pitch
        self.E_max = max_energy
        self.sigma = scale_sigma

    def compute_toroidal_helix(
        self,
        theta: np.ndarray,
        phase_offset: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Calculates 3D coordinates along a helical torus trajectory.

        X(theta) = (R + r * cos(k * theta + phi)) * cos(theta)
        Y(theta) = (R + r * cos(k * theta + phi)) * sin(theta)
        Z(theta) = r * sin(k * theta + phi)

        Args:
            theta: Array of toroidal angles [0, 2*pi]
            phase_offset: Phase shift phi (rad)

        Returns:
            Tuple (X, Y, Z) coordinate arrays
        """
        psi = self.k * theta + phase_offset
        X = (self.R + self.r * np.cos(psi)) * np.cos(theta)
        Y = (self.R + self.r * np.cos(psi)) * np.sin(theta)
        Z = self.r * np.sin(psi)
        return X, Y, Z

    def compute_phase_diff(self, phase_a: float, phase_b: float) -> float:
        """
        Computes normalized minimal angular phase difference in [-pi, pi].
        """
        diff = (phase_a - phase_b + np.pi) % (2.0 * np.pi) - np.pi
        return float(diff)

    def compute_gate_barrier_energy(self, phase_diff: float) -> float:
        """
        Calculates the barrier tension energy E_gate(Delta phi).
        E_gate = E_max * tanh(|Delta phi| / sigma)

        When Delta phi -> 0, E_gate -> 0 (Relaxation, Gate Opens).
        When Delta phi != 0, E_gate > 0 (Barrier maintains resistance).
        """
        normalized_diff = abs(phase_diff)
        return float(self.E_max * np.tanh(normalized_diff / self.sigma))

    def compute_permeability(self, phase_diff: float) -> float:
        """
        Calculates gate permeability P_gate in [0.0, 1.0].
        P_gate = 1.0 - E_gate / E_max = 1.0 - tanh(|Delta phi| / sigma)
        """
        e_gate = self.compute_gate_barrier_energy(phase_diff)
        return float(max(0.0, min(1.0, 1.0 - (e_gate / self.E_max))))

    def evaluate_linking_gate(
        self,
        ring_phase_a: float,
        ring_phase_b: float,
        tolerance: float = 1e-3
    ) -> Dict[str, Any]:
        """
        Evaluates topological linking condition between two torus rings.

        Returns:
            Dict containing phase_diff, energy, permeability, and is_linked status.
        """
        phase_diff = self.compute_phase_diff(ring_phase_a, ring_phase_b)
        energy = self.compute_gate_barrier_energy(phase_diff)
        permeability = self.compute_permeability(phase_diff)
        is_linked = bool(abs(phase_diff) <= tolerance or energy < 1e-4)

        return {
            "phase_diff": phase_diff,
            "gate_energy": energy,
            "permeability": permeability,
            "is_linked": is_linked
        }


class CryptographicPhaseMatcher:
    """
    Cryptographic Phase Matching & Topological Decryption Engine.

    Translates payload cryptographic data into prescribed phase offsets
    and evaluates phase key alignment to open topological gates.
    """

    def __init__(self, gate: Optional[HelicalOffsetTorusGate] = None):
        self.gate = gate or HelicalOffsetTorusGate()

    def encode_payload_to_phase(self, payload_hash: int) -> float:
        """
        Maps an integer or hash payload to a prescribed cipher phase offset in [-pi, pi].
        """
        # Map integer hash mod 2*pi to phase
        normalized = (payload_hash % 1000000) / 1000000.0
        phase = normalized * 2.0 * np.pi - np.pi
        return float(phase)

    def decrypt_topological_gate(
        self,
        cipher_phase: float,
        key_phase: float,
        payload_data: Any = None,
        key_tolerance: float = 0.05
    ) -> Dict[str, Any]:
        """
        Attempts decryption by matching key phase against cipher phase offset.

        If key phase aligns with cipher phase, gate barrier tension drops to 0,
        enabling non-destructive data passage (Decryption).
        """
        phase_diff = self.gate.compute_phase_diff(cipher_phase, key_phase)
        energy = self.gate.compute_gate_barrier_energy(phase_diff)
        permeability = self.gate.compute_permeability(phase_diff)

        success = bool(abs(phase_diff) <= key_tolerance and permeability > 0.9)

        return {
            "success": success,
            "phase_diff": phase_diff,
            "barrier_energy": energy,
            "permeability": permeability,
            "decrypted_payload": payload_data if success else None
        }
