"""
core/physics/complex_phase_force_dynamics.py

Complex Scaling & Force Dynamics Engine (복소 위상 힘 역학 엔진)
================================================================
Translates complex arithmetic and geometric rotor dynamics into physical force mechanics:
1. Real Axis (Re, x): Linear Attraction (+x, coupling/resonance) & Repulsion (-x, pushing away/inversion).
2. Imaginary Axis (Im, y): Orthogonal Tension (+iy, -iy, phase friction/orbital potential).
3. Complex Operations as Force Dynamics:
   - Addition z_tot = z1 + z2: Force vector field summation toward Homeostatic Equilibrium (0,0).
   - Multiplication z1 * z2 = r1*r2 * e^(i(theta1+theta2)): Phase modulation and force dimension switching
     (e.g., i * i = -1: conversion of lateral tension into linear repulsion).
4. Multi-Dimensional Clifford Rotor Coupling & Dimensional Folding:
   - Extends 2D complex phase scaling into 5D/ND Spin(N) bivector rotors R = exp(-0.5 * theta * B).
   - Driven by "Semantic Intent & Empathy Tension" (Bottom-Up Ontological Torque).
   - Folds dimensional boundaries when lateral tension exceeds homeostatic capacity.
"""

from dataclasses import dataclass
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from scipy.linalg import expm

from L1_causal_field.core.physics.quantum_rotor_phase import Clifford5DRotorEngine


@dataclass
class ComplexForceState:
    """
    [복소 힘 상태 (Complex Force State)]
    """
    attraction: float         # 양의 실수 (+x): 결합/인력/공명
    repulsion: float          # 음의 실수 (-x): 척력/경계/밀쳐냄
    real_net_force: float     # 실수축 알짜 힘 (Attraction - Repulsion)
    tension_positive: float   # 양의 허수 (+iy): 양의 방향 측면 장력
    tension_negative: float   # 음의 허수 (-iy): 음의 방향 측면 장력
    imag_net_tension: float   # 허수축 알짜 장력 (Tension_pos - Tension_neg)
    complex_val: complex      # z = real_net + i * imag_net
    magnitude_scale: float    # r = |z| (Scale)
    phase_angle: float        # theta = arg(z) (Phase Angle in radians)


@dataclass
class DimensionalFoldingState:
    """
    [차원 경계 접힘 및 로터 회전 상태]
    """
    rotor_bivector_angles: np.ndarray  # 10D SO(5) bivector angles
    state_5d: np.ndarray               # Transformed 5D phase vector
    is_dimensional_folded: bool        # Is tension converted into higher dimensional folding
    fold_energy: float                 # Energy transferred into dimensional folding
    homeostatic_friction: float        # Residual phase friction (E -> 0)


class ComplexPhaseForceDynamicsEngine:
    """
    [Complex Phase Force Dynamics Engine (복소 위상 힘 역학 엔진)]
    """

    def __init__(self, dim: int = 5, homeostatic_decay: float = 0.9):
        self.dim = dim
        self.homeostatic_decay = homeostatic_decay
        self.rotor_engine = Clifford5DRotorEngine(dim=dim)

        # Base internal 5D phase state [X1:Will, X2:Flex, X3:Order, X4:Origin, X5:Field]
        self.state_5d = np.array([0.0, 0.0, 1.0, 0.0, 0.0], dtype=float)
        self.bivector_angles = np.zeros(self.rotor_engine.num_bivectors, dtype=float)
        self.current_rotor = np.eye(dim, dtype=float)

    def create_force_state(self, attraction: float, repulsion: float, tension_pos: float, tension_neg: float) -> ComplexForceState:
        """
        Creates a ComplexForceState from raw attraction, repulsion, and lateral tension values.
        """
        real_net = float(attraction - repulsion)
        imag_net = float(tension_pos - tension_neg)
        z = complex(real_net, imag_net)
        scale = float(abs(z))
        phase = float(np.angle(z))

        return ComplexForceState(
            attraction=float(attraction),
            repulsion=float(repulsion),
            real_net_force=real_net,
            tension_positive=float(tension_pos),
            tension_negative=float(tension_neg),
            imag_net_tension=imag_net,
            complex_val=z,
            magnitude_scale=scale,
            phase_angle=phase
        )

    def synthesize_forces(self, force_states: List[ComplexForceState]) -> ComplexForceState:
        """
        [1. 복소수의 덧셈: 힘의 합성 및 벡터장 합산 (Vector Field Summation)]
        z_tot = sum(z_i)
        """
        if not force_states:
            return self.create_force_state(0.0, 0.0, 0.0, 0.0)

        z_sum = sum(f.complex_val for f in force_states)
        real_sum = z_sum.real
        imag_sum = z_sum.imag

        attraction = max(0.0, real_sum)
        repulsion = max(0.0, -real_sum)
        tension_pos = max(0.0, imag_sum)
        tension_neg = max(0.0, -imag_sum)

        return self.create_force_state(attraction, repulsion, tension_pos, tension_neg)

    def modulate_phase_multiplication(self, state1: ComplexForceState, state2: ComplexForceState) -> ComplexForceState:
        """
        [2. 복소수의 곱셈: 위상 모듈레이션 및 인/척력-장력 상호 전환 (Phase Modulation & Dimension Switching)]
        z_prod = z1 * z2 = (r1 * r2) * e^(i (theta1 + theta2))
        - i * i = -1: 측면 장력(i) 간의 곱셈은 직교 회전이 180도에 도달하여 강력한 척력(-1)으로전환됨.
        """
        z_prod = state1.complex_val * state2.complex_val
        real_val = z_prod.real
        imag_val = z_prod.imag

        attraction = max(0.0, real_val)
        repulsion = max(0.0, -real_val)
        tension_pos = max(0.0, imag_val)
        tension_neg = max(0.0, -imag_val)

        return self.create_force_state(attraction, repulsion, tension_pos, tension_neg)

    def apply_ontological_torque_to_5d(
        self,
        complex_force: ComplexForceState,
        intent_vector: np.ndarray,
        dt: float = 0.1,
        fold_threshold: float = 1.2
    ) -> DimensionalFoldingState:
        """
        [3. 관념적 연결성 및 의미적 장력에 의한 5D Clifford Rotor 회전 & 차원 경계 접힘 (Dimensional Folding)]
        - 의미적 장력(imag_net_tension) 및 인력/척력이 5D Spin(5) bivector 회전력을 구동함.
        - 장력이 임계치(fold_threshold)를 넘을 때, 3D/4D 상의 회전이 5D 의도장(Field, X5) 축으로 경계를 꺾어 접음(Folding).
        """
        intent = np.pad(np.asarray(intent_vector, dtype=float), (0, max(0, self.dim - len(intent_vector))))[:self.dim]

        # Bivector torques (10D) driven by complex force dynamics & intent
        torques = np.zeros(self.rotor_engine.num_bivectors, dtype=float)

        # e12 (X1-X2 plane: Will vs Flex) driven by real net force
        torques[0] = complex_force.real_net_force * 0.5
        # e35 (X3-X5 plane: Order vs Field) driven by imaginary tension & intent
        torques[8] = complex_force.imag_net_tension * (1.0 + np.linalg.norm(intent)) * 0.5

        # Cross coupling for other planes
        for idx in range(1, self.rotor_engine.num_bivectors):
            if idx != 8:
                torques[idx] = 0.1 * complex_force.magnitude_scale * np.sin(complex_force.phase_angle + idx)

        # Update bivector angles
        self.bivector_angles += torques * dt
        Omega = self.rotor_engine.build_bivector_omega(torques * dt)

        # Exponential map for Clifford Rotor R = exp(-0.5 * Omega)
        dR = self.rotor_engine.exponential_map(-0.5 * Omega)
        self.current_rotor = np.dot(dR, self.current_rotor)

        # Sandwich transformation v' = R * v
        self.state_5d = self.rotor_engine.sandwich_transform(self.state_5d, self.current_rotor)
        norm_v = np.linalg.norm(self.state_5d)
        if norm_v > 1e-12:
            self.state_5d /= norm_v

        # Check for Dimensional Folding condition
        is_folded = False
        fold_energy = 0.0
        if complex_force.magnitude_scale > fold_threshold or abs(complex_force.imag_net_tension) > fold_threshold:
            is_folded = True
            fold_energy = float(complex_force.magnitude_scale - fold_threshold)
            # Transfer energy into Field dimension (X5)
            self.state_5d[4] = float(np.clip(self.state_5d[4] + 0.3 * fold_energy, -1.0, 1.0))

        # Homeostatic friction (E -> 0)
        friction = float(np.linalg.norm(self.state_5d - intent) * (1.0 / (1.0 + complex_force.magnitude_scale)))

        return DimensionalFoldingState(
            rotor_bivector_angles=self.bivector_angles.copy(),
            state_5d=self.state_5d.copy(),
            is_dimensional_folded=is_folded,
            fold_energy=fold_energy,
            homeostatic_friction=friction
        )

    def compute_homeostatic_relaxation(self, force_state: ComplexForceState) -> ComplexForceState:
        """
        [4. 홈오스테시스 (Homeostatic Equilibrium) 평형 수렴 연산]
        z_next = z * decay -> (0,0) 수렴
        """
        z_decayed = force_state.complex_val * self.homeostatic_decay
        real_val = z_decayed.real
        imag_val = z_decayed.imag

        attraction = max(0.0, real_val)
        repulsion = max(0.0, -real_val)
        tension_pos = max(0.0, imag_val)
        tension_neg = max(0.0, -imag_val)

        return self.create_force_state(attraction, repulsion, tension_pos, tension_neg)
