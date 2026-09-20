"""
Human Cognitive Phase Dynamics Module for Elysia Engine.

Implements the 4-Phase Human Cognitive Development Framework and
Multi-Scale Cross-Frequency Phase Coupling (CFC) Engine.

1. Dual Phase Coupling:
   - Slow Phase (theta_i ~ 4-12 Hz): Macro-context, intuitive trajectory, global gating.
   - Fast Phase (phi_i ~ 30-80 Hz): Micro-computation, local feature resonance, symbolic grounding.
2. Differential Dynamics:
   - d(theta_i)/dt = omega_slow + K_slow * sum_j( exp(-d_ij) * sin(theta_j - theta_i) ) + alpha_fb * sin(phi_i - theta_i)
   - d(phi_i)/dt   = omega_fast + M_mod * cos(theta_i) + K_fast * sum_j( exp(-d_ij) * sin(phi_j - phi_i) ) + K_ext * sin(Phi_ext - phi_i)
3. Hebbian Phase Plasticity:
   - Dynamic topological manifold deformation d(i, j) based on phase co-resonance.
4. Meta-Cognition & Edge of Chaos (Criticality) Control:
   - Dynamic adjustment of coupling parameter K to balance rigid phase-locking and chaos.
5. Hierarchical Attractors (1st-order Sensory Attractors -> 2nd-order Meta Attractors).
"""

import math
import time
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class MultiScalePhaseCouplingEngine:
    """
    Multi-Scale Cross-Frequency Phase Coupling (CFC) Engine.
    Simulates dual-scale phase dynamics for N interconnected concept/sensory nodes.
    """

    def __init__(
        self,
        num_nodes: int = 64,
        dt: float = 0.005,
        K_slow: float = 0.15,
        K_fast: float = 0.30,
        M_mod: float = 2.00,
        alpha_fb: float = 0.08,
        use_torch: bool = False
    ):
        self.num_nodes = num_nodes
        self.dt = dt
        self.K_slow = K_slow
        self.K_fast = K_fast
        self.M_mod = M_mod
        self.alpha_fb = alpha_fb
        self.use_torch = use_torch and HAS_TORCH

        # Initial phases in [0, 2*pi)
        self.slow_phase = np.random.uniform(0.0, 2.0 * np.pi, num_nodes).astype(np.float32)
        self.fast_phase = np.random.uniform(0.0, 2.0 * np.pi, num_nodes).astype(np.float32)

        # Natural frequencies (Slow: Theta band ~6Hz, Fast: Gamma band ~40Hz)
        self.slow_omega = np.random.normal(6.0, 0.5, num_nodes).astype(np.float32)
        self.fast_omega = np.random.normal(40.0, 2.0, num_nodes).astype(np.float32)

        # Causal Metric Distance Matrix d_ij (default initial uniform topological metric)
        coords = np.linspace(0.0, 10.0, num_nodes, dtype=np.float32)
        self.metric_dist = np.abs(coords[:, None] - coords[None, :]) + 0.1
        self.metric_dist = np.ascontiguousarray(self.metric_dist, dtype=np.float32)

    def set_metric_distance(self, distance_matrix: np.ndarray) -> None:
        """Sets custom causal distance matrix N x N."""
        assert distance_matrix.shape == (self.num_nodes, self.num_nodes)
        self.metric_dist = np.ascontiguousarray(distance_matrix, dtype=np.float32)

    def step(
        self,
        external_sensory_signal: Optional[np.ndarray] = None,
        custom_dt: Optional[float] = None,
        coupling_gain: float = 10.0
    ) -> Dict[str, Any]:
        """
        Executes one Euler integration step of multi-scale cross-frequency phase coupling.
        """
        dt = custom_dt if custom_dt is not None else self.dt
        N = self.num_nodes

        if external_sensory_signal is None:
            ext_drive = np.zeros(N, dtype=np.float32)
        else:
            ext_sig = np.array(external_sensory_signal, dtype=np.float32)
            if len(ext_sig) < N:
                ext_sig = np.pad(ext_sig, (0, N - len(ext_sig)))
            elif len(ext_sig) > N:
                ext_sig = ext_sig[:N]

            # Physical Kuramoto drive: coupling_gain * sin(Phi_ext - phi_i) for non-zero signals
            ext_drive = coupling_gain * np.sin(ext_sig - self.fast_phase) * (np.abs(ext_sig) > 1e-4)

        # Spatial interaction weight: exp(-d_ij)
        spatial_influence = np.exp(-self.metric_dist)  # N x N

        # 1. Spatial Phase Interactions
        slow_diff = self.slow_phase[None, :] - self.slow_phase[:, None]
        fast_diff = self.fast_phase[None, :] - self.fast_phase[:, None]

        sum_slow_interaction = np.sum(spatial_influence * np.sin(slow_diff), axis=1)
        sum_fast_interaction = np.sum(spatial_influence * np.sin(fast_diff), axis=1)

        # 2. Cross-Frequency Coupling (CFC)
        # Top-down gating: slow phase modulates fast phase velocity
        top_down_gating = self.M_mod * np.cos(self.slow_phase)
        # Bottom-up feedback: fast phase convergence feeds back to slow phase
        bottom_up_feedback = self.alpha_fb * np.sin(self.fast_phase - self.slow_phase)

        # 3. Phase Derivatives
        d_slow_dt = self.slow_omega + (self.K_slow * sum_slow_interaction) + bottom_up_feedback
        d_fast_dt = self.fast_omega + top_down_gating + (self.K_fast * sum_fast_interaction) + ext_drive

        # 4. Euler Integration & 2*pi modulo clamping
        TWO_PI = 2.0 * np.pi
        self.slow_phase = (self.slow_phase + d_slow_dt * dt + TWO_PI) % TWO_PI
        self.fast_phase = (self.fast_phase + d_fast_dt * dt + TWO_PI) % TWO_PI

        # 5. Compute Order Parameters (R, Psi) for Slow and Fast Oscillators
        z_slow = np.mean(np.exp(1j * self.slow_phase))
        z_fast = np.mean(np.exp(1j * self.fast_phase))

        r_slow = float(np.abs(z_slow))
        psi_slow = float(np.angle(z_slow))

        r_fast = float(np.abs(z_fast))
        psi_fast = float(np.angle(z_fast))

        delta_phi_fast = float(1.0 - r_fast)
        delta_phi_slow = float(1.0 - r_slow)

        return {
            "slow_phase": self.slow_phase.copy(),
            "fast_phase": self.fast_phase.copy(),
            "order_R_slow": r_slow,
            "order_psi_slow": psi_slow,
            "order_R_fast": r_fast,
            "order_psi_fast": psi_fast,
            "delta_phi_fast": delta_phi_fast,
            "delta_phi_slow": delta_phi_slow,
            "top_down_gating_mean": float(np.mean(top_down_gating)),
            "bottom_up_feedback_mean": float(np.mean(bottom_up_feedback))
        }


class HebbianPhasePlasticity:
    """
    Hebbian Phase Plasticity Engine.
    Dynamically deforms the topological metric distance matrix d(i, j) based on
    co-resonance across slow and fast phase waves.
    """

    def __init__(
        self,
        num_nodes: int = 64,
        plasticity_rate: float = 0.05,
        decay_rate: float = 0.005,
        min_dist: float = 0.05,
        max_dist: float = 15.0
    ):
        self.num_nodes = num_nodes
        self.lr = plasticity_rate
        self.decay = decay_rate
        self.min_dist = min_dist
        self.max_dist = max_dist

    def update_metric(
        self,
        current_metric: np.ndarray,
        slow_phase: np.ndarray,
        fast_phase: np.ndarray,
        default_metric: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Updates d_ij: Nodes that oscillate in phase pull closer (d_ij decreases),
        while un-resonated connections slowly decay towards default state.
        """
        if default_metric is None:
            default_metric = current_metric

        slow_cos = np.cos(slow_phase[None, :] - slow_phase[:, None])
        fast_cos = np.cos(fast_phase[None, :] - fast_phase[:, None])

        # Dual-scale joint resonance factor in [-1.0, 1.0]
        joint_resonance = 0.4 * slow_cos + 0.6 * fast_cos

        # Co-resonant nodes (resonance > 0) reduce distance d_ij
        delta_d = -self.lr * joint_resonance + self.decay * (default_metric - current_metric)

        updated_metric = current_metric + delta_d
        # Ensure symmetric distance with zero diagonal
        updated_metric = 0.5 * (updated_metric + updated_metric.T)
        updated_metric = np.clip(updated_metric, self.min_dist, self.max_dist)
        np.fill_diagonal(updated_metric, 0.0)

        return updated_metric.astype(np.float32)


class MetaCognitiveCriticalityGovernor:
    """
    Meta-Cognitive Governor managing the Edge of Chaos (Criticality).
    Monitors global convergence error Delta Phi and dynamically adjusts
    coupling parameters K and noise to prevent rigid fixation or phase chaos.
    """

    def __init__(
        self,
        target_delta_phi_range: Tuple[float, float] = (0.10, 0.45),
        k_min: float = 0.05,
        k_max: float = 3.0,
        adaptation_rate: float = 0.05
    ):
        self.target_min, self.target_max = target_delta_phi_range
        self.k_min = k_min
        self.k_max = k_max
        self.adaptation_rate = adaptation_rate
        self.cognitive_state = "EQUILIBRIUM"  # "EQUILIBRIUM", "RIGID_FIXATION", "CHAOTIC_DISPERSION", "ACTIVE_LEARNING"

    def adapt_criticality(
        self,
        engine: MultiScalePhaseCouplingEngine,
        delta_phi_fast: float,
        recalled_attractor_found: bool = True
    ) -> Dict[str, Any]:
        """
        Evaluates cognitive state and adjusts K_fast and K_slow.
        """
        old_k_fast = engine.K_fast

        if delta_phi_fast < self.target_min:
            # Rigid fixation / Over-locking -> Lower K to allow flexible thinking
            self.cognitive_state = "RIGID_FIXATION_PREJUDICE"
            engine.K_fast = float(max(self.k_min, engine.K_fast - self.adaptation_rate))
            engine.K_slow = float(max(self.k_min, engine.K_slow - 0.5 * self.adaptation_rate))
            # Add slight phase noise to break lock
            engine.fast_phase += np.random.normal(0, 0.1, engine.num_nodes).astype(np.float32)

        elif delta_phi_fast > self.target_max:
            if not recalled_attractor_found:
                # High error and no known attractor -> Active learning / Novel stimulus exploration
                self.cognitive_state = "ACTIVE_NOVEL_CONCEPT_LEARNING"
                # Keep K moderate to allow forming a new attractor
                engine.K_fast = float(np.clip(engine.K_fast + 0.5 * self.adaptation_rate, self.k_min, self.k_max))
            else:
                # Excessive chaos -> Increase K to force synchronization
                self.cognitive_state = "CHAOTIC_DISPERSION"
                engine.K_fast = float(min(self.k_max, engine.K_fast + self.adaptation_rate))
                engine.K_slow = float(min(self.k_max, engine.K_slow + 0.5 * self.adaptation_rate))
        else:
            self.cognitive_state = "OPTIMAL_CRITICALITY_FLEXIBLE_COGNITION"
            # Gently nudge towards baseline K
            engine.K_fast = float(np.clip(engine.K_fast, 0.2, 1.0))

        return {
            "cognitive_state": self.cognitive_state,
            "delta_phi_fast": delta_phi_fast,
            "old_K_fast": old_k_fast,
            "new_K_fast": engine.K_fast,
            "new_K_slow": engine.K_slow
        }


def _get_relative_phase(phase_array: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Computes mean-subtracted relative phase geometry invariant to uniform rotation."""
    if mask is not None and np.sum(mask) > 0:
        z_mean = np.sum(np.exp(1j * phase_array) * mask)
    else:
        z_mean = np.mean(np.exp(1j * phase_array))
    mean_angle = np.angle(z_mean) if np.abs(z_mean) > 1e-6 else 0.0
    return (phase_array - mean_angle + np.pi) % (2.0 * np.pi) - np.pi


class HierarchicalAttractorNetwork:
    """
    Hierarchical Attractor Causal Memory.
    Supports 1st-Order Sensory Concept Attractors and 2nd-Order Meta Attractors.
    Invariant to uniform phase rotation via relative phase geometry matching.
    """

    def __init__(self, num_nodes: int = 64, resonance_threshold: float = 0.50):
        self.num_nodes = num_nodes
        self.resonance_threshold = resonance_threshold
        self.first_order_attractors: List[Dict[str, Any]] = []
        self.second_order_attractors: List[Dict[str, Any]] = []

    def store_1st_order_attractor(
        self,
        label: str,
        fast_phase_state: np.ndarray,
        concept_metadata: Optional[Dict[str, Any]] = None,
        active_mask: Optional[np.ndarray] = None
    ) -> int:
        """Stores a primary sensory concept attractor (e.g. 'Apple' = red + round)."""
        attractor_id = len(self.first_order_attractors)
        rel_state = _get_relative_phase(fast_phase_state, active_mask)
        self.first_order_attractors.append({
            "id": attractor_id,
            "order": 1,
            "label": label,
            "phase_state": fast_phase_state.copy(),
            "relative_state": rel_state,
            "metadata": concept_metadata or {},
            "access_count": 0
        })
        return attractor_id

    def store_2nd_order_meta_attractor(
        self,
        meta_label: str,
        child_attractor_ids: List[int],
        meta_slow_phase_state: np.ndarray,
        meta_metadata: Optional[Dict[str, Any]] = None
    ) -> int:
        """Stores a high-level meta attractor (e.g. 'Fruit' = Apple, Banana, Grape phase relation)."""
        meta_id = len(self.second_order_attractors)
        rel_state = _get_relative_phase(meta_slow_phase_state)
        self.second_order_attractors.append({
            "id": meta_id,
            "order": 2,
            "label": meta_label,
            "child_ids": child_attractor_ids,
            "phase_state": meta_slow_phase_state.copy(),
            "relative_state": rel_state,
            "metadata": meta_metadata or {},
            "access_count": 0
        })
        return meta_id

    def recall_1st_order(self, current_fast_phase: np.ndarray, active_mask: Optional[np.ndarray] = None) -> Optional[Dict[str, Any]]:
        """Recalls closest 1st-order sensory concept attractor based on relative phase resonance."""
        if not self.first_order_attractors:
            return None

        best_match = None
        highest_resonance = -1.0

        current_rel = _get_relative_phase(current_fast_phase, active_mask)

        for attr in self.first_order_attractors:
            diff_cos = np.cos(current_rel - attr["relative_state"])
            if active_mask is not None and np.sum(active_mask) > 0:
                resonance = float(np.sum(diff_cos * active_mask) / np.sum(active_mask))
            else:
                resonance = float(np.mean(diff_cos))

            if resonance > highest_resonance:
                highest_resonance = resonance
                best_match = attr

        if highest_resonance >= self.resonance_threshold and best_match is not None:
            best_match["access_count"] += 1
            return {
                "attractor": best_match,
                "resonance": highest_resonance
            }
        return None

    def recall_2nd_order(self, current_slow_phase: np.ndarray) -> Optional[Dict[str, Any]]:
        """Recalls closest 2nd-order meta concept attractor based on relative slow-phase macro resonance."""
        if not self.second_order_attractors:
            return None

        best_match = None
        highest_resonance = -1.0

        current_rel = _get_relative_phase(current_slow_phase)

        for attr in self.second_order_attractors:
            resonance = float(np.mean(np.cos(current_rel - attr["relative_state"])))
            if resonance > highest_resonance:
                highest_resonance = resonance
                best_match = attr

        if highest_resonance >= self.resonance_threshold and best_match is not None:
            best_match["access_count"] += 1
            return {
                "attractor": best_match,
                "resonance": highest_resonance
            }
        return None
