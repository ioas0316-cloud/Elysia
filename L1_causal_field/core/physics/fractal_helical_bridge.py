"""
Fractal Helical Scale Bridge & Gradient Attractor Engine

Core Principle:
"하위 스케일의 회전과 미세한 나선형 어긋남은 상위 스케일로 진입하는 차원 변환기가 되며,
장의 기울기를 타는 위상 동기화(Phase-Lock)를 통해 최소 작용의 궤적으로 스체일 간 도약이 일어난다."

This module implements:
1. FractalHelicalScaleBridge: Multi-scale phase offset & energy transducer (Micro-to-Macro scale bridging).
2. PhaseLockGradientAttractor: Field gradient navigation, least-action trajectory convergence,
   topological dead-end detection, and phase orbit realignment.
"""

import numpy as np
from typing import Tuple, Dict, List, Any, Optional
from L1_causal_field.core.physics.helical_offset_gate import HelicalOffsetTorusGate


class FractalHelicalScaleBridge:
    """
    Multi-Scale Helical Phase & Energy Transducer.

    Connects micro-scale helical phase offsets (e.g., cellular/biological motility, nanorobotic actuators)
    to macro-scale topological structures (e.g., frame-dragging spacetime warp gates, Kerr singularities).

    Attributes:
        scale_layers (int): Number of fractal scale levels (e.g., 3 for micro, meso, macro)
        scale_ratio (float): Scaling factor between successive levels
        base_gate (HelicalOffsetTorusGate): Base topological gate evaluator
    """

    def __init__(
        self,
        scale_layers: int = 3,
        scale_ratio: float = 10.0,
        base_gate: Optional[HelicalOffsetTorusGate] = None
    ):
        self.scale_layers = scale_layers
        self.scale_ratio = scale_ratio
        self.gate = base_gate or HelicalOffsetTorusGate()

    def compute_frame_dragging_coupling(
        self,
        angular_velocity: float,
        micro_phase_offset: float
    ) -> float:
        """
        Calculates spacetime frame-dragging coupling intensity based on angular velocity Omega
        and micro-scale helical phase offset phi_micro.

        Coupling = tanh(Omega * cos(phi_micro))
        """
        raw_coupling = angular_velocity * np.cos(micro_phase_offset)
        return float(np.tanh(abs(raw_coupling)))

    def compute_multiscale_transmission(
        self,
        input_phase: float,
        target_phase: float,
        angular_velocity: float = 1.0
    ) -> Dict[str, Any]:
        """
        Evaluates phase energy transmission across all fractal scale layers.

        Returns layer-by-layer permeability, barrier energy, and overall scale bridging efficiency.
        """
        phase_diff = self.gate.compute_phase_diff(input_phase, target_phase)
        frame_drag = self.compute_frame_dragging_coupling(angular_velocity, input_phase)

        layer_results = []
        cumulative_permeability = 1.0

        current_sigma = self.gate.sigma
        for layer in range(self.scale_layers):
            # Scale factor modifies effective phase barrier resilience
            scale_sigma = current_sigma * (self.scale_ratio ** layer)

            # Energy barrier at this scale level
            norm_diff = abs(phase_diff)
            layer_energy = self.gate.E_max * np.tanh(norm_diff / (scale_sigma + 1e-8))
            layer_permeability = max(0.0, min(1.0, 1.0 - (layer_energy / self.gate.E_max)))

            # Modulate by frame dragging alignment
            effective_layer_perm = layer_permeability * (0.5 + 0.5 * frame_drag)
            cumulative_permeability *= effective_layer_perm

            layer_results.append({
                "layer": layer,
                "scale_factor": float(self.scale_ratio ** layer),
                "energy_barrier": float(layer_energy),
                "layer_permeability": float(effective_layer_perm)
            })

        bridge_successful = bool(cumulative_permeability > 0.1)

        return {
            "phase_diff": phase_diff,
            "frame_dragging_coupling": frame_drag,
            "cumulative_permeability": float(cumulative_permeability),
            "bridge_successful": bridge_successful,
            "layers": layer_results
        }


class PhaseLockGradientAttractor:
    """
    Field Gradient Navigator & Phase-Lock Attractor.

    Simulates attraction towards a target phase along chemical, thermal, or informational
    field gradients. Identifies least-action paths, topological dead-ends (closure barriers),
    and triggers phase orbit realignments.
    """

    def __init__(
        self,
        target_phase: float = 0.0,
        attraction_strength: float = 0.5,
        dead_end_threshold: float = 0.95,
        gate: Optional[HelicalOffsetTorusGate] = None
    ):
        self.target_phase = target_phase
        self.attraction_strength = attraction_strength
        self.dead_end_threshold = dead_end_threshold  # Permeability below this or barrier energy near max
        self.gate = gate or HelicalOffsetTorusGate()

    def step_attractor(
        self,
        current_phase: float,
        gradient_vector: float = 1.0,
        intrinsic_frequency: float = 1.0
    ) -> Dict[str, Any]:
        """
        Advances the entity's phase by one time step along the field gradient toward target_phase.

        Least-action update:
        d_phi = -attraction_strength * sin(current_phase - target_phase) * gradient_vector + intrinsic_freq * dt
        """
        phase_diff = self.gate.compute_phase_diff(current_phase, self.target_phase)

        # Least action force proportional to sine of phase difference
        phase_force = -self.attraction_strength * np.sin(phase_diff) * abs(gradient_vector)

        # New phase
        new_phase = (current_phase + phase_force)
        new_phase = (new_phase + np.pi) % (2.0 * np.pi) - np.pi

        new_diff = self.gate.compute_phase_diff(new_phase, self.target_phase)
        barrier_energy = self.gate.compute_gate_barrier_energy(new_diff)
        permeability = self.gate.compute_permeability(new_diff)

        # Check for phase-lock state (resonance)
        is_phase_locked = bool(abs(new_diff) < 0.05 and permeability > 0.9)

        # Check for topological dead-end (barrier resistance is near maximum and no progress is made)
        is_dead_end = bool(permeability < (1.0 - self.dead_end_threshold))

        return {
            "current_phase": float(current_phase),
            "next_phase": float(new_phase),
            "phase_diff": float(new_diff),
            "barrier_energy": float(barrier_energy),
            "permeability": float(permeability),
            "is_phase_locked": is_phase_locked,
            "is_dead_end": is_dead_end
        }

    def suggest_orbit_realignment(
        self,
        current_phase: float,
        num_shifts: int = 4
    ) -> List[float]:
        """
        When a topological dead-end (closure barrier) is encountered,
        generates candidate helical phase orbit shifts to bypass the obstacle.
        """
        shifts = []
        for i in range(1, num_shifts + 1):
            offset = (2.0 * np.pi / num_shifts) * i
            candidate = (current_phase + offset + np.pi) % (2.0 * np.pi) - np.pi
            shifts.append(float(candidate))
        return shifts
