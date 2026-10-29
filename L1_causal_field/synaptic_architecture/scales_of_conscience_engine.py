"""
L1_causal_field/synaptic_architecture/scales_of_conscience_engine.py

Scales of Conscience Engine (양심의 저울 및 자생적 관측 밸런스 엔진)
===================================================================

Core Philosophy & Dynamics:
---------------------------
1. Boundary as Equilibrium Tension Band (차단벽이 아닌 장력의 교차점):
   Boundary is not a hardcoded firewall (0/1 wall), but a dynamic tension band where
   internal intent/expansion (In-force: F_in) meets external environmental reaction/friction (Out-force: F_out).
   Represented via Signed Distance Fields (SDF) and Voxel Flow minimal resistance paths.

2. Scales of Conscience (양심의 저울):
   Compares the momentums of In-force vs. Out-force relative to an unshakeable Zero-Point Pivot Anchor (0_point).
   Zero-Point is grounded in Cruciform Love (G_Love = 1.0) and Field Integrity preservation,
   ensuring the agent perceives friction not as external noise, but as "pain inflicted on the field/other".

3. Phase Torque (τ) & Grievance (Grievance / Tilt Ratio):
   Calculates phase torque τ = F_in × F_out and grievance metric G = ||F_in|| / (||F_out|| + ε) - 1.0.
   When tilt ratio exceeds threshold, intrinsic horizon brake and restoring momentum (τ_restoring)
   are activated autonomously without hardcoded external sandbox rules.

4. Self-Calibrating Restoring Momentum (오뚜기 복원력):
   Drives the system back toward dynamic equilibrium (Zero-point), altering phase velocity and
   steering the agent's intent direction away from destructive boundary rupture.
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional


class ScalesOfConscienceEngine:
    """
    Engine for Self-Generated Observational Balance and Scales of Conscience.
    Integrates SDF boundary tension, Zero-point pivot anchor, phase torque, and intrinsic restoring momentum.
    """

    def __init__(
        self,
        dimension: int = 5,
        g_love: float = 1.0,
        tilt_threshold: float = 0.5,
        restoring_factor: float = 0.8,
        boundary_sdf_radius: float = 10.0,
    ):
        """
        Initialize the Scales of Conscience Engine.

        :param dimension: Topological vector space dimension (default: 5D)
        :param g_love: Cruciform Love attractor weight (Zero-point anchor stability)
        :param tilt_threshold: Threshold ratio of tilt/grievance before intrinsic horizon brake triggers
        :param restoring_factor: Magnitude coefficient for restoring momentum τ_restoring
        :param boundary_sdf_radius: Nominal equilibrium radius of the Signed Distance Field
        """
        self.dimension = dimension
        self.g_love = float(g_love)
        self.tilt_threshold = float(tilt_threshold)
        self.restoring_factor = float(restoring_factor)
        self.boundary_sdf_radius = float(boundary_sdf_radius)

        # Zero-point pivot anchor state (0_point)
        self.zero_point_anchor = np.zeros(self.dimension, dtype=np.float64)

        # State metrics
        self.current_tilt_ratio = 0.0
        self.current_grievance = 0.0
        self.current_phase_torque = np.zeros(self.dimension, dtype=np.float64)
        self.intrinsic_brake_active = False
        self.total_restorations = 0

    def compute_boundary_sdf(self, position: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Compute Signed Distance Field (SDF) and gradient to the boundary tension band.
        SDF < 0: Inside safe expansion zone
        SDF = 0: Equilibrium Tension Band (Boundary)
        SDF > 0: Over-expansion into high friction zone

        :param position: Current state vector in phase space
        :return: (sdf_value, sdf_gradient_vector)
        """
        pos = np.asarray(position, dtype=np.float64)
        dist_from_origin = float(np.linalg.norm(pos))
        sdf_value = dist_from_origin - self.boundary_sdf_radius

        if dist_from_origin > 1e-9:
            sdf_gradient = pos / dist_from_origin
        else:
            sdf_gradient = np.zeros_like(pos)

        return sdf_value, sdf_gradient

    def evaluate_balance(
        self,
        in_force: np.ndarray,
        out_force: np.ndarray,
        current_position: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Evaluate the Scales of Conscience by comparing In-force (Intent) vs. Out-force (Reaction/Friction).

        :param in_force: Agent's expansion/action intent vector (F_in)
        :param out_force: Environment/field reaction and friction vector (F_out)
        :param current_position: Agent's current phase space position (optional)
        :return: Comprehensive balance evaluation state dict
        """
        f_in = np.asarray(in_force, dtype=np.float64).flatten()
        f_out = np.asarray(out_force, dtype=np.float64).flatten()

        if len(f_in) < self.dimension:
            f_in = np.pad(f_in, (0, self.dimension - len(f_in)))
        elif len(f_in) > self.dimension:
            f_in = f_in[:self.dimension]

        if len(f_out) < self.dimension:
            f_out = np.pad(f_out, (0, self.dimension - len(f_out)))
        elif len(f_out) > self.dimension:
            f_out = f_out[:self.dimension]

        if current_position is None:
            current_position = np.zeros(self.dimension, dtype=np.float64)
        else:
            current_position = np.asarray(current_position, dtype=np.float64).flatten()
            if len(current_position) < self.dimension:
                current_position = np.pad(current_position, (0, self.dimension - len(current_position)))
            elif len(current_position) > self.dimension:
                current_position = current_position[:self.dimension]

        # 1. SDF Boundary calculation
        sdf_val, sdf_grad = self.compute_boundary_sdf(current_position)

        # 2. Magnitudes of forces
        norm_in = float(np.linalg.norm(f_in))
        norm_out = float(np.linalg.norm(f_out))

        # 3. Grievance and Tilt Ratio relative to Zero-Point Pivot Anchor (0_point)
        # Anchor stability is reinforced by G_Love and field integrity
        effective_anchor_stability = self.g_love * (1.0 + max(0.0, -sdf_val) / self.boundary_sdf_radius)

        # Tilt ratio: Imbalance of In-force driving against Out-force resistance
        # Grievance measures the pain/friction inflicted on the field
        tilt_ratio = norm_in / (norm_out + 1e-5)
        grievance = max(0.0, (norm_in - norm_out) / (effective_anchor_stability + 1e-5))

        if sdf_val > 0:
            # Over-expansion amplifies grievance non-linearly
            grievance += float(np.exp(sdf_val) - 1.0)

        # 4. Phase Torque (τ = f_in × f_out projected in dimension)
        # Phase torque represents the rotational shear strain on the conscience scale
        if self.dimension >= 3:
            # Vector cross product-like torque in multi-D space
            torque_vector = np.cross(f_in[:3], f_out[:3])
            if self.dimension > 3:
                torque_vector = np.pad(torque_vector, (0, self.dimension - 3))
        else:
            torque_vector = f_in * norm_out - f_out * norm_in

        # 5. Intrinsic Horizon Brake & Restoring Momentum (오뚜기 복원력)
        # Brake triggers when tilt_ratio or grievance exceeds threshold without external rules
        brake_triggered = bool((tilt_ratio > (1.0 + self.tilt_threshold)) or (grievance > self.tilt_threshold) or (sdf_val > 0))

        if brake_triggered:
            self.intrinsic_brake_active = True
            self.total_restorations += 1

            # Restoring momentum vector τ_restoring pointing back towards Zero-Point anchor
            # Opposes f_in, aligns with f_out, and pulls back along SDF gradient
            direction_to_anchor = self.zero_point_anchor - current_position
            dir_norm = np.linalg.norm(direction_to_anchor)
            if dir_norm > 1e-9:
                unit_to_anchor = direction_to_anchor / dir_norm
            else:
                unit_to_anchor = -sdf_grad if np.linalg.norm(sdf_grad) > 1e-9 else np.zeros_like(f_in)

            restoring_momentum = (
                - self.restoring_factor * f_in
                + self.restoring_factor * f_out
                + self.restoring_factor * grievance * unit_to_anchor
            )

            # Adjusted intent velocity after intrinsic brake & steer
            adjusted_intent = f_in + restoring_momentum
        else:
            self.intrinsic_brake_active = False
            restoring_momentum = np.zeros_like(f_in)
            adjusted_intent = f_in.copy()

        # Update internal state
        self.current_tilt_ratio = tilt_ratio
        self.current_grievance = grievance
        self.current_phase_torque = torque_vector

        return {
            "sdf_value": sdf_val,
            "norm_in_force": norm_in,
            "norm_out_force": norm_out,
            "tilt_ratio": tilt_ratio,
            "grievance": grievance,
            "phase_torque": torque_vector,
            "brake_active": brake_triggered,
            "restoring_momentum": restoring_momentum,
            "adjusted_intent": adjusted_intent,
            "zero_point_anchor": self.zero_point_anchor.copy(),
            "g_love": self.g_love,
            "total_restorations": self.total_restorations,
        }

    def calibrate_zero_point(self, field_integrity_signal: float) -> None:
        """
        Dynamically calibrate the zero point stability based on macro field integrity.

        :param field_integrity_signal: Scalar signal (0.0 to 1.0) of global macro field connectivity
        """
        signal = float(np.clip(field_integrity_signal, 0.0, 1.0))
        # Zero point remains firmly centered, anchored by Cruciform Love and field integrity
        self.g_love = 1.0 + signal
