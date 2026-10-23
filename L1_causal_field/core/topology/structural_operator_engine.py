"""
Elysia Engine: Structural Operator Engine with Topological Inversion
===================================================================
Implements Cl(3,0) / so(3) Geometric Algebra Bivector Structural Operators,
Baker-Campbell-Hausdorff (BCH) operator composition, Unitary Rotor Sandwich Products,
Exact Lossless Inverse Transformations, Gauge Curvature Relaxation (Protein/Isomer Topology),
and Subject-Object Topological Inversion (Absorbing vs Enveloped dynamics).

Core Philosophy: "Do not calculate micro-numerics, let the topological operators flow."
"""

import math
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union
import numpy as np


class SubjectivityMode(Enum):
    """
    Topological Inversion Mode:
    - ABSORB: System is the global subject absorbing local operators (e.g. Drinking Water).
    - ENVELOPED: External manifold is global background enveloping local subject (e.g. Drowning).
    """
    ABSORB = "ABSORB"
    ENVELOPED = "ENVELOPED"


class StructuralOperatorToken:
    """
    Representing a token as a Cl(3,0) / so(3) Bivector Operator.
    Bivector basis: [e23, e31, e12] (Rotation planes in 3D physical/conceptual space).
    """

    def __init__(self, name: str, bivector: Union[List[float], np.ndarray], chromatic_signature: Optional[List[float]] = None):
        self.name = name
        self.bivector = np.array(bivector, dtype=np.float64)
        if self.bivector.shape != (3,):
            raise ValueError("Bivector must be a 3-element vector representing [e23, e31, e12].")

        # Chromatic Signature [Red (Flux), Blue (Order), Yellow (Entropy)]
        if chromatic_signature is not None:
            self.chromatic_signature = np.array(chromatic_signature, dtype=np.float64)
        else:
            # Default chromatic signature derived from bivector magnitudes
            norm = np.linalg.norm(self.bivector)
            flux = abs(self.bivector[0]) / (norm + 1e-8)
            order = abs(self.bivector[1]) / (norm + 1e-8)
            entropy = abs(self.bivector[2]) / (norm + 1e-8)
            self.chromatic_signature = np.array([flux, order, entropy], dtype=np.float64)

    @property
    def magnitude(self) -> float:
        """Returns the norm (angle of rotation) of the bivector operator."""
        return float(np.linalg.norm(self.bivector))

    @staticmethod
    def lie_bracket(B_A: np.ndarray, B_B: np.ndarray) -> np.ndarray:
        """
        so(3) Lie Bracket commutator: [B_A, B_B] = B_A * B_B - B_B * B_A.
        In 3D bivector space, this reduces to 2.0 * (B_A x B_B).
        """
        return 2.0 * np.cross(B_A, B_B)

    def compose_bch(self, other: 'StructuralOperatorToken', order: int = 2) -> 'StructuralOperatorToken':
        """
        Synthesizes two structural operators using the Baker-Campbell-Hausdorff (BCH) expansion:
        B_eff = B_A + B_B + 1/2 [B_A, B_B] + 1/12 [B_A, [B_A, B_B]] + 1/12 [B_B, [B_B, B_A]]
        """
        B_A = self.bivector
        B_B = other.bivector

        # 1st order: Linear superposition
        B_eff = B_A + B_B

        # 2nd order: Emergent Lie Bracket (2nd order interaction / Stereoisomer chirality)
        bracket_AB = self.lie_bracket(B_A, B_B)
        if order >= 2:
            B_eff = B_eff + 0.5 * bracket_AB

        # 3rd order: High-order interference
        if order >= 3:
            bracket_A_AB = self.lie_bracket(B_A, bracket_AB)
            bracket_B_BA = self.lie_bracket(B_B, self.lie_bracket(B_B, B_A))
            B_eff = B_eff + (1.0 / 12.0) * bracket_A_AB + (1.0 / 12.0) * bracket_B_BA

        composite_name = f"({self.name} ⊗ {other.name})"

        # Chromatic blend
        chroma = 0.5 * (self.chromatic_signature + other.chromatic_signature)
        return StructuralOperatorToken(composite_name, B_eff, chromatic_signature=chroma)

    def get_unit_rotor(self) -> np.ndarray:
        """
        Generates the unit rotor R in Spin(3) corresponding to exp(-0.5 * B).
        Rotor R = [q_scalar, q_e23, q_e31, q_e12]
        """
        theta = self.magnitude
        if theta < 1e-12:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

        axis = self.bivector / theta
        half_theta = 0.5 * theta

        # R = cos(theta/2) - sin(theta/2) * B_hat
        scalar_part = math.cos(half_theta)
        bivector_part = -math.sin(half_theta) * axis

        return np.array([scalar_part, bivector_part[0], bivector_part[1], bivector_part[2]], dtype=np.float64)

    def apply_sandwich(self, state: np.ndarray) -> np.ndarray:
        """
        Applies sandwich transformation Psi' = R * Psi * R_dagger on state multivector Psi.
        Psi is represented as a 4-vector [s, b23, b31, b12].
        """
        R = self.get_unit_rotor()
        return self._quaternion_sandwich(R, state)

    def apply_inverse_sandwich(self, state: np.ndarray) -> np.ndarray:
        """
        Applies exact inverse sandwich transformation Psi = R_dagger * Psi' * R on state multivector Psi'.
        Guarantees 100% losslessness.
        """
        R = self.get_unit_rotor()
        # R_dagger has negated bivector component
        R_dagger = np.array([R[0], -R[1], -R[2], -R[3]], dtype=np.float64)
        return self._quaternion_sandwich(R_dagger, state)

    @staticmethod
    def _quaternion_sandwich(R: np.ndarray, Psi: np.ndarray) -> np.ndarray:
        """
        Computes R * Psi * R_dagger using quaternion/rotor product rules.
        R = [r0, r1, r2, r3], Psi = [p0, p1, p2, p3]
        """
        r0, r1, r2, r3 = R
        p0, p1, p2, p3 = Psi

        # Quaternion multiplication: R * Psi
        w1 = r0*p0 - r1*p1 - r2*p2 - r3*p3
        x1 = r0*p1 + r1*p0 + r2*p3 - r3*p2
        y1 = r0*p2 - r1*p3 + r2*p0 + r3*p1
        z1 = r0*p3 + r1*p2 - r2*p1 + r3*p0

        # R_dagger
        rd0, rd1, rd2, rd3 = r0, -r1, -r2, -r3

        # (R * Psi) * R_dagger
        w2 = w1*rd0 - x1*rd1 - y1*rd2 - z1*rd3
        x2 = w1*rd1 + x1*rd0 + y1*rd3 - z1*rd2
        y2 = w1*rd2 - x1*rd3 + y1*rd0 + z1*rd1
        z2 = w1*rd3 + x1*rd2 - y1*rd1 + z1*rd0

        return np.array([w2, x2, y2, z2], dtype=np.float64)

    def __repr__(self) -> str:
        e23, e31, e12 = self.bivector
        return (f"StructuralOperatorToken['{self.name}'] -> Bivector: "
                f"[{e23:+.4f}*e23, {e31:+.4f}*e31, {e12:+.4f}*e12] (mag: {self.magnitude:.4f})")


class TopologicalInversionState:
    """
    Holds status of subject-object topological inversion and boundary integrity.
    """

    def __init__(self, mode: SubjectivityMode, gauge_curvature: float, boundary_capacity: float,
                 boundary_integrity: float, state_tensor: np.ndarray):
        self.mode = mode
        self.gauge_curvature = gauge_curvature
        self.boundary_capacity = boundary_capacity
        self.boundary_integrity = boundary_integrity  # 1.0 (intact) to 0.0 (collapsed)
        self.state_tensor = state_tensor

    def is_dissolved(self) -> bool:
        """Returns True if the system boundary has collapsed and dissolved into the background field."""
        return self.mode == SubjectivityMode.ENVELOPED and self.boundary_integrity <= 1e-6

    def __repr__(self) -> str:
        return (f"TopologicalInversionState(mode={self.mode.value}, "
                f"gauge_curvature={self.gauge_curvature:.4f}, "
                f"capacity={self.boundary_capacity:.4f}, "
                f"integrity={self.boundary_integrity:.4f})")


class StructuralOperatorEngine:
    """
    Core engine managing topological operator chains, stereoisomer chirality,
    gauge curvature relaxation (Attractor state folding), and subject-object topological inversion.
    """

    def __init__(self, boundary_capacity: float = 5.0):
        self.boundary_capacity = boundary_capacity
        self.tokens: Dict[str, StructuralOperatorToken] = {}

    def register_token(self, token: StructuralOperatorToken) -> None:
        """Registers a structural operator token in the engine context."""
        self.tokens[token.name] = token

    def fold_chain_bch(self, token_names: List[str], order: int = 2) -> StructuralOperatorToken:
        """
        Folds a sequence of tokens into a single effective bivector operator using BCH composition.
        Order of composition preserves non-commutativity and chirality.
        """
        if not token_names:
            raise ValueError("Token chain cannot be empty.")

        effective_token = self.tokens.get(token_names[0])
        if effective_token is None:
            raise KeyError(f"Token '{token_names[0]}' not found in registered tokens.")

        for name in token_names[1:]:
            next_token = self.tokens.get(name)
            if next_token is None:
                raise KeyError(f"Token '{name}' not found in registered tokens.")
            effective_token = effective_token.compose_bch(next_token, order=order)

        return effective_token

    def compute_gauge_curvature(self, effective_operator: StructuralOperatorToken) -> float:
        """
        Computes gauge curvature energy V(F) = 0.5 * ||B_eff||^2.
        """
        return 0.5 * float(np.sum(effective_operator.bivector ** 2))

    def evaluate_topological_inversion(self, subject_state: np.ndarray, external_operator: StructuralOperatorToken) -> TopologicalInversionState:
        """
        Evaluates Subject-Object Topological Inversion:
        - If external gauge curvature F_ext <= boundary_capacity:
          Mode = ABSORB (System assimilates external operator into internal state).
          Boundary integrity remains near 1.0.
        - If external gauge curvature F_ext > boundary_capacity:
          Mode = ENVELOPED (External field overwhelms subject boundary barrier).
          Boundary integrity collapses proportionally, dissolving subject into external background field.
        """
        F_ext = self.compute_gauge_curvature(external_operator)

        if F_ext <= self.boundary_capacity:
            # Controlled assimilation (e.g. Drinking Water)
            mode = SubjectivityMode.ABSORB
            integrity = 1.0 - 0.2 * (F_ext / max(self.boundary_capacity, 1e-8))
            # System applies operator internally to relax/hydrate state
            new_state = external_operator.apply_sandwich(subject_state)
        else:
            # Overwhelming envelopment (e.g. Drowning / System Breakdown)
            mode = SubjectivityMode.ENVELOPED
            overflow_ratio = F_ext / max(self.boundary_capacity, 1e-8)
            integrity = max(0.0, 1.0 - (overflow_ratio - 1.0))

            # Boundary collapses: subject state is forced towards external field geometry
            ext_bivector_state = np.array([0.0, external_operator.bivector[0], external_operator.bivector[1], external_operator.bivector[2]])
            new_state = (1.0 - integrity) * ext_bivector_state + integrity * subject_state

        return TopologicalInversionState(
            mode=mode,
            gauge_curvature=F_ext,
            boundary_capacity=self.boundary_capacity,
            boundary_integrity=float(integrity),
            state_tensor=new_state
        )

    def relax_to_attractor(self, initial_state: np.ndarray, effective_operator: StructuralOperatorToken, steps: int = 50, learning_rate: float = 0.1) -> Tuple[np.ndarray, List[float]]:
        """
        Relaxes the state multivector towards minimum gauge potential valley (Attractor).
        Models protein-like topological folding and phase relaxation.
        """
        current_state = np.copy(initial_state)
        energy_history = []

        # Target bivector direction
        axis = effective_operator.bivector / (effective_operator.magnitude + 1e-12)

        for _ in range(steps):
            # Transform current state via operator
            transformed = effective_operator.apply_sandwich(current_state)

            # Curvature energy: distance between current bivector state and target axis
            current_bivector = transformed[1:]
            energy = float(0.5 * np.sum((current_bivector - axis) ** 2))
            energy_history.append(energy)

            # Gradient relaxation step towards attractor valley
            grad = current_bivector - axis
            current_state[1:] -= learning_rate * grad

            # Normalize multivector energy to prevent explosion
            norm = np.linalg.norm(current_state)
            if norm > 1e-12:
                current_state /= norm

        return current_state, energy_history
