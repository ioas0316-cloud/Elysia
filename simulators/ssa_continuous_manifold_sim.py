"""
SSA Continuous Manifold Simulator (자발적 구조 적응 시뮬레이터)
===================================================================

This module provides a full simulation loop for demonstrating the Self-Supervised Structural Adaptation (SSA) engine.
It simulates continuous wave fields entering through multiple lenses, phase friction energy minimization,
modified Ricci flow metric adaptation, and the emergence of self-carved volumetric manifold tokens.

Key Features:
- Multi-step temporal evolution with dynamic external wave perturbation.
- Tracking of Phase Friction Energy, Mean Scalar Curvature, Active Lens Selection, and Emergent Volumetric Tokens.
- Console status dashboard and structured JSON step log generator for visualization/analysis.
"""

import math
import time
import json
from typing import Dict, List, Any, Optional

import numpy as np

from synaptic_architecture.self_supervised_structural_adaptation import (
    SelfSupervisedStructuralAdaptationEngine,
    ContinuousSensoryField,
    InternalManifoldState,
    MultiLensProjectionOperator
)


class SSAContinuousManifoldSimulator:
    """
    Simulation harness for running continuous sensory wave inputs through the SSA engine.
    """
    def __init__(
        self,
        grid_size: int = 16,
        kuramoto_coupling_K: float = 1.5,
        learning_rate_phase: float = 0.2,
        learning_rate_metric: float = 0.05,
        resonance_threshold: float = 0.08
    ):
        self.grid_size = grid_size
        self.engine = SelfSupervisedStructuralAdaptationEngine(
            grid_size=grid_size,
            kuramoto_coupling_K=kuramoto_coupling_K,
            learning_rate_phase=learning_rate_phase,
            learning_rate_metric=learning_rate_metric,
            resonance_threshold=resonance_threshold
        )
        self.history: List[Dict[str, Any]] = []

    def run_simulation(
        self,
        num_steps: int = 50,
        wave_freq: float = 1.0,
        time_delta: float = 0.1,
        verbose: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Runs the simulation loop for a specified number of time steps.
        """
        if verbose:
            print("=" * 70)
            print("  STARTING SSA CONTINUOUS MANIFOLD SIMULATION")
            print(f"  Grid Size: {self.grid_size}x{self.grid_size} | Steps: {num_steps} | Δt: {time_delta}")
            print("=" * 70)

        for t_idx in range(num_steps):
            t_current = t_idx * time_delta

            # Generate continuous external wave field S(x, t) with dynamic frequency modulation
            freq_mod = wave_freq + 0.2 * math.sin(t_current * 0.5)
            sensory_field = ContinuousSensoryField.generate_wave(
                spatial_dim=self.grid_size,
                t=t_current,
                freq=freq_mod,
                phase_shift=0.1 * t_current
            )

            # Execute SSA Loop Step
            step_res = self.engine.step(sensory_field, time_delta=time_delta)

            record = {
                "step": t_idx + 1,
                "time_t": round(t_current, 3),
                "wave_freq": round(freq_mod, 3),
                "friction_energy": round(float(step_res["friction_energy"]), 6),
                "active_lens": step_res["active_lens"],
                "mean_scalar_curvature": round(float(step_res["mean_scalar_curvature"]), 6),
                "carved_volumes_count": step_res["carved_volumes_count"],
                "carved_volumes": step_res["carved_volumes"]
            }
            self.history.append(record)

            if verbose and ((t_idx + 1) % 5 == 0 or t_idx == 0 or t_idx == num_steps - 1):
                print(
                    f"[Step {t_idx+1:03d} | t={t_current:5.2f}s] "
                    f"Friction: {record['friction_energy']:.6f} | "
                    f"Lens: {record['active_lens']:12s} | "
                    f"R_scalar: {record['mean_scalar_curvature']:.6f} | "
                    f"Tokens: {record['carved_volumes_count']}"
                )

        if verbose:
            print("=" * 70)
            print("  SIMULATION COMPLETE")
            print(f"  Initial Friction: {self.history[0]['friction_energy']:.6f} -> Final Friction: {self.history[-1]['friction_energy']:.6f}")
            print(f"  Total Carved Volumetric Tokens Emerged: {self.history[-1]['carved_volumes_count']}")
            print("=" * 70)

        return self.history

    def export_history_json(self, filepath: str = "ssa_simulation_log.json"):
        """Exports full simulation history to JSON file."""
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.history, f, indent=2, ensure_ascii=False)
        print(f"Simulation log successfully written to {filepath}")


if __name__ == "__main__":
    simulator = SSAContinuousManifoldSimulator(grid_size=16)
    simulator.run_simulation(num_steps=30, verbose=True)
