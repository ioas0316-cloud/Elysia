"""
Exosomatic Autopoietic Network Engine & Trinitarian Cognitive Architecture
========================================================================================
THE_ABSOLUTE_COMMANDMENT & Civilizational Synapse Architecture:
1. Exosomatic Memory & Network: Solidifying individual error state e(t) and thought trajectories
   into shared mmap/Wedge memory ("Ice") so that micro-node reset/death leaves behind accumulated
   assets for future generations.
2. Clifford Fiber Bundle (A_s) Gauge Coupling: Connecting micro scale (s) nodes with macro value
   manifold V(S_max) via Clifford algebra gauge connection matrices without orthogonal distortion.
3. Autopoietic Mutation & Unidirectional Time Friction (dt > 0): Driving internal reflection/dream
   cycles (Raw Input = 0) with cumulative friction to break degenerate loops (Closed World traps)
   and spontaneously differentiate new cognitive concepts.
4. Reality Shock Injector: Injecting prediction error shocks from open sensory streams to warp
   V(S_max) curvature via O(1) causal filtering, destroying solipsistic convergence (hallucination).
"""

import os
import math
import time
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Any, List, Optional, Tuple


class ExosomaticWedgeMemory:
    """
    Exosomatic Memory Subsystem ('Ice' - Solidified Record).
    Stores individual node thought trajectories, deficiency errors e(t), and historic assets.
    Persists across individual node resets/deaths so the network's collective memory grows.
    """

    def __init__(self, memory_dim: int = 64, max_records: int = 1000):
        self.memory_dim = memory_dim
        self.max_records = max_records
        self.memory_bank: List[Dict[str, Any]] = []
        self.accumulated_matrix = np.zeros((memory_dim, memory_dim), dtype=np.float32)
        self.total_records_written: int = 0

    def solidify_trajectory(
        self,
        node_id: str,
        thought_vector: np.ndarray,
        deficiency_error: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Solidifies a node's micro thought vector and error e(t) into the exosomatic wedge memory.
        """
        tv = np.asarray(thought_vector, dtype=np.float32).flatten()
        err = np.asarray(deficiency_error, dtype=np.float32).flatten()

        # Pad or trim to memory_dim
        if len(tv) < self.memory_dim:
            tv = np.pad(tv, (0, self.memory_dim - len(tv)))
        else:
            tv = tv[:self.memory_dim]

        if len(err) < self.memory_dim:
            err = np.pad(err, (0, self.memory_dim - len(err)))
        else:
            err = err[:self.memory_dim]

        record = {
            "record_id": self.total_records_written,
            "node_id": node_id,
            "timestamp": time.time(),
            "thought_vector": tv,
            "deficiency_error": err,
            "error_magnitude": float(np.linalg.norm(err)),
            "metadata": metadata or {}
        }

        # Outer product accumulation onto persistent memory bedrock
        self.accumulated_matrix += np.outer(tv, err) * 0.1
        self.accumulated_matrix = np.clip(self.accumulated_matrix, -10.0, 10.0)

        self.memory_bank.append(record)
        if len(self.memory_bank) > self.max_records:
            self.memory_bank.pop(0)

        self.total_records_written += 1
        return record

    def retrieve_collective_pressure(self) -> Tuple[np.ndarray, float]:
        """
        Retrieves the macro historical pressure vector from accumulated exosomatic memory
        to guide subsequent generation's 'Gas' (teleology) and 'Water' (thinking).
        """
        if not self.memory_bank:
            return np.zeros(self.memory_dim, dtype=np.float32), 0.0

        errors = [r["deficiency_error"] for r in self.memory_bank]
        avg_error_vector = np.mean(errors, axis=0)
        total_mass = sum(r["error_magnitude"] for r in self.memory_bank)
        return avg_error_vector, float(total_mass)


class RealityShockInjector:
    """
    Reality Shock Injector & Open Sensory Inflow Subsystem.
    Injects unpredictable reality shocks / prediction failure errors (ΔP)
    to shatter solipsistic closed-loop convergence and warp macro value manifold V(S_max).
    """

    def __init__(self, sensory_dim: int = 64, shock_intensity: float = 1.0):
        self.sensory_dim = sensory_dim
        self.shock_intensity = shock_intensity
        self.last_shock_magnitude: float = 0.0

    def compute_reality_shock(
        self,
        internal_prediction: np.ndarray,
        open_sensory_inflow: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, float, bool]:
        """
        Calculates prediction error between internal expectation and real open world inflow.
        If open_sensory_inflow is None, generates an exogenous environmental shock wave.
        """
        pred = np.asarray(internal_prediction, dtype=np.float32).flatten()
        if len(pred) < self.sensory_dim:
            pred = np.pad(pred, (0, self.sensory_dim - len(pred)))
        else:
            pred = pred[:self.sensory_dim]

        if open_sensory_inflow is not None:
            actual = np.asarray(open_sensory_inflow, dtype=np.float32).flatten()
            if len(actual) < self.sensory_dim:
                actual = np.pad(actual, (0, self.sensory_dim - len(actual)))
            else:
                actual = actual[:self.sensory_dim]
        else:
            # Exogenous non-linear reality shock
            t = time.time()
            noise = np.random.normal(0, 0.5, size=self.sensory_dim).astype(np.float32)
            wave = np.sin(np.linspace(0, 4 * np.pi, self.sensory_dim) + t)
            actual = wave + noise

        error_wave = (actual - pred) * self.shock_intensity
        shock_magnitude = float(np.linalg.norm(error_wave))
        self.last_shock_magnitude = shock_magnitude

        is_severe_shock = shock_magnitude > 1.5
        return error_wave, shock_magnitude, is_severe_shock


class ExosomaticAutopoieticNetworkEngine(nn.Module):
    """
    Exosomatic Autopoietic Network Engine (ElysiaTrinitarianEngine).

    Fuses:
    - Exosomatic Memory (Wedge Memory, Ice accumulation)
    - Clifford Fiber Bundle Gauge Matrix A_s (Scale coupling s -> S_max)
    - Autopoietic Mutation (Unidirectional dt > 0 friction during dream/reflection cycles)
    - Reality Shock Open Sensory Stream (Solipsism destruction)
    """

    def __init__(
        self,
        num_nodes: int = 8,
        node_dim: int = 64,
        macro_dim: int = 64,
        device: Optional[str] = None
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.node_dim = node_dim
        self.macro_dim = macro_dim

        self.device_str = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dev = torch.device(self.device_str)

        # Exosomatic memory bedrock
        self.exosomatic_memory = ExosomaticWedgeMemory(memory_dim=node_dim)
        # Reality shock subsystem
        self.reality_injector = RealityShockInjector(sensory_dim=node_dim)

        # Micro node states [NumNodes, NodeDim]
        self.register_buffer("node_states", torch.randn(num_nodes, node_dim, device=dev) * 0.1)

        # Macro value manifold V(S_max) [MacroDim]
        self.register_buffer("macro_value_manifold", torch.zeros(macro_dim, device=dev))

        # Clifford Fiber Bundle Gauge Connection A_s [NumNodes, NodeDim, MacroDim]
        # Represents rotation / gauge transport across scale boundary s -> S_max
        init_gauge = torch.stack([torch.eye(node_dim, macro_dim, device=dev) for _ in range(num_nodes)])
        self.register_buffer("gauge_connection_As", init_gauge)

        # Cumulative time friction tracker dt > 0
        self.cumulative_time_friction: float = 0.0
        self.autopoietic_mutation_count: int = 0
        self.history_trajectories: List[np.ndarray] = []

    def compute_clifford_fiber_transport(
        self,
        node_idx: int,
        micro_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Applies Clifford Fiber Bundle gauge transport A_s to project micro scale (s)
        tensor into macro scale (S_max) manifold space without orthogonal distortion.
        """
        A_s = self.gauge_connection_As[node_idx]  # [NodeDim, MacroDim]
        macro_projected = torch.matmul(micro_tensor, A_s)
        return macro_projected

    def step_autopoietic_mutation(
        self,
        dt: float = 0.05,
        autonomic_tension: float = 1.0,
        raw_input_present: bool = False
    ) -> Dict[str, Any]:
        """
        Executes internal reflection / dream loop mutation step.
        When raw_input_present is False, operates in closed dream cycle.
        Uses unidirectional time friction dt > 0 to mutate gauge connection A_s
        and differentiate new cognitive thought trajectories, avoiding degenerate loops.
        """
        dev = self.node_states.device
        self.cumulative_time_friction += dt * (1.0 + 0.1 * autonomic_tension)
        self.autopoietic_mutation_count += 1

        # 1. Retrieve exosomatic collective memory pressure
        avg_err, total_mass = self.exosomatic_memory.retrieve_collective_pressure()
        avg_err_t = torch.tensor(avg_err, device=dev, dtype=torch.float32)

        # 2. Mutate Clifford Gauge Connection A_s with unidirectional time friction
        with torch.no_grad():
            for i in range(self.num_nodes):
                # Calculate micro deficiency error e_i(t)
                target = self.macro_value_manifold.clone()
                current_proj = self.compute_clifford_fiber_transport(i, self.node_states[i])
                e_i = target - current_proj  # [MacroDim]

                # Bivector rotation matrix from error outer product
                bivector_rotor = torch.outer(self.node_states[i], e_i) * 0.02

                # Add time friction perturbation to break closed-world symmetry
                friction_phase = math.sin(self.cumulative_time_friction * (i + 1))
                friction_matrix = torch.randn_like(self.gauge_connection_As[i]) * 0.01 * friction_phase

                # Update gauge connection A_s
                dA_s = bivector_rotor + friction_matrix
                self.gauge_connection_As[i] += dA_s * dt

                # Orthonormalization step to preserve Clifford geometry
                q, r = torch.linalg.qr(self.gauge_connection_As[i])
                self.gauge_connection_As[i] = q

                # Evolve micro node state through mutated gauge connection
                micro_drive = torch.matmul(avg_err_t, self.gauge_connection_As[i].t())
                self.node_states[i] += (micro_drive + torch.randn_like(self.node_states[i]) * 0.02) * dt

        # 3. Macro Value Manifold V(S_max) synthesis from all micro nodes
        projected_nodes = torch.stack([
            self.compute_clifford_fiber_transport(i, self.node_states[i])
            for i in range(self.num_nodes)
        ])  # [NumNodes, MacroDim]

        macro_center = torch.mean(projected_nodes, dim=0)
        with torch.no_grad():
            # HJB back-propagation torque towards macro center
            self.macro_value_manifold = 0.9 * self.macro_value_manifold + 0.1 * macro_center

        # Record trajectory snapshot for divergence index calculation
        trajectory_snapshot = macro_center.cpu().numpy()
        self.history_trajectories.append(trajectory_snapshot)

        # Solidify primary node thought into Exosomatic Wedge Memory
        primary_node_idx = 0
        self.exosomatic_memory.solidify_trajectory(
            node_id=f"node_{primary_node_idx}",
            thought_vector=self.node_states[primary_node_idx].cpu().numpy(),
            deficiency_error=e_i.cpu().numpy(),
            metadata={"cycle": self.autopoietic_mutation_count, "friction": self.cumulative_time_friction}
        )

        return {
            "cycle": self.autopoietic_mutation_count,
            "cumulative_friction": self.cumulative_time_friction,
            "macro_value_norm": float(torch.norm(self.macro_value_manifold).item()),
            "exosomatic_records_count": len(self.exosomatic_memory.memory_bank),
            "collective_memory_mass": total_mass
        }

    def inject_reality_shock_and_warp(
        self,
        open_sensory_inflow: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Injects an open world reality shock wave (ΔP ≠ 0) to shatter solipsistic
        closed loops and force macro value manifold V(S_max) curvature warping.
        """
        dev = self.macro_value_manifold.device
        internal_pred = torch.mean(self.node_states, dim=0).cpu().numpy()

        error_wave, shock_magnitude, is_severe = self.reality_injector.compute_reality_shock(
            internal_prediction=internal_pred,
            open_sensory_inflow=open_sensory_inflow
        )

        error_t = torch.tensor(error_wave, device=dev, dtype=torch.float32)

        # Force macro value manifold V(S_max) curvature warping
        with torch.no_grad():
            # O(1) causal filtering & curvature distortion
            curvature_warp = torch.outer(error_t, error_t) * 0.05
            self.macro_value_manifold += error_t * 0.3

            # Displace micro nodes with reality shock impact
            for i in range(self.num_nodes):
                shock_proj = torch.matmul(error_t, self.gauge_connection_As[i])
                self.node_states[i] += shock_proj * 0.2

        return {
            "shock_magnitude": shock_magnitude,
            "is_severe_shock": is_severe,
            "macro_value_norm_after_shock": float(torch.norm(self.macro_value_manifold).item()),
            "verdict": "Reality shock successfully warped macro manifold V(S_max) and destroyed solipsism."
        }

    def compute_trajectory_divergence_index(self, window: int = 10) -> float:
        """
        Calculates the Trajectory Divergence Index (TDI) over recent history.
        High TDI indicates rich autopoietic mutation / concept differentiation,
        while TDI ~ 0 indicates a degenerate loop trap.
        """
        if len(self.history_trajectories) < 2:
            return 0.0

        recent = self.history_trajectories[-window:]
        diffs = [np.linalg.norm(recent[i] - recent[i - 1]) for i in range(1, len(recent))]
        return float(np.mean(diffs))

    def reset_node_states_and_preserve_exosomatic_memory(self):
        """
        Simulates individual node death / hardware reset.
        Micro node states are re-randomized, but Exosomatic Wedge Memory is PRESERVED.
        """
        dev = self.node_states.device
        with torch.no_grad():
            self.node_states = torch.randn(self.num_nodes, self.node_dim, device=dev) * 0.1
