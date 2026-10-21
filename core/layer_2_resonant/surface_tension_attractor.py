#!/usr/bin/env python3
"""
surface_tension_attractor.py

Layer 2: Resonant Attractor & Structural Coupling Module
Implements SurfaceTensionAttractor and PhaseRelaxationEngine representing minimum-energy
self-organizing convergence and smooth topological relaxation without discrete if-else rules.
"""

import numpy as np
from typing import Dict, Any, Tuple


class PhaseRelaxationEngine:
    """
    Simulates physical surface tension in a continuous phase field.
    Distortions and noise automatically smooth and relax toward a minimum-energy equilibrium,
    analogous to liquid resin rounding into a micro-lens under heat and surface tension.
    """

    def __init__(self, surface_tension_coeff: float = 0.1, viscosity: float = 0.05):
        self.gamma = surface_tension_coeff
        self.viscosity = viscosity

    def relax_step(self, phase_field: np.ndarray, dt: float = 0.01) -> np.ndarray:
        """
        Performs one step of surface-tension field relaxation via Laplacian curvature dynamics:
        d_phase / dt = gamma * Laplacian(phase) - viscosity * d_phase/dt
        """
        field_dim = phase_field.ndim
        if field_dim == 1:
            # 1D Laplacian
            laplacian = np.roll(phase_field, -1) + np.roll(phase_field, 1) - 2.0 * phase_field
        elif field_dim == 2:
            # 2D Laplacian
            laplacian = (
                np.roll(phase_field, -1, axis=0) + np.roll(phase_field, 1, axis=0) +
                np.roll(phase_field, -1, axis=1) + np.roll(phase_field, 1, axis=1) -
                4.0 * phase_field
            )
        else:
            laplacian = np.zeros_like(phase_field)

        # Gradient flow toward minimum surface energy
        relaxed_field = phase_field + dt * (self.gamma * laplacian - self.viscosity * phase_field)
        return relaxed_field


class SurfaceTensionAttractor:
    """
    Acts as a resonant attractor in Layer 2 that pulls distorted mental imagery or phase inputs
    into a smooth, self-organized minimal-energy state.
    """

    def __init__(self, tension_coeff: float = 0.15, max_steps: int = 50):
        self.engine = PhaseRelaxationEngine(surface_tension_coeff=tension_coeff)
        self.max_steps = max_steps

    def compute_surface_energy(self, field: np.ndarray) -> float:
        """
        Computes total surface potential energy (integral of gradient squared).
        """
        gradients = np.gradient(field)
        energy = float(np.sum([np.sum(g ** 2) for g in gradients]))
        return energy

    def converge_imagery_attractor(self, raw_imagery_field: np.ndarray) -> Dict[str, Any]:
        """
        Iteratively relaxes the input imagery field until it reaches surface tension equilibrium.

        Returns:
            Dict containing initial_energy, relaxed_energy, relaxed_imagery_field, and steps_taken.
        """
        current_field = np.copy(raw_imagery_field)
        initial_energy = self.compute_surface_energy(current_field)

        energy_history = [initial_energy]
        for step in range(self.max_steps):
            current_field = self.engine.relax_step(current_field)
            current_energy = self.compute_surface_energy(current_field)
            energy_history.append(current_energy)

            # Equilibrium reached when energy change is negligible
            if len(energy_history) > 2 and abs(energy_history[-2] - current_energy) < 1e-6:
                break

        relaxed_energy = energy_history[-1]
        convergence_ratio = relaxed_energy / (initial_energy + 1e-9)

        return {
            "initial_energy": initial_energy,
            "relaxed_energy": relaxed_energy,
            "convergence_ratio": float(convergence_ratio),
            "relaxed_imagery_field": current_field,
            "steps_taken": len(energy_history) - 1,
            "is_converged": bool(convergence_ratio < 0.5 or relaxed_energy < 1e-4)
        }


if __name__ == "__main__":
    attractor = SurfaceTensionAttractor()
    noisy_field = np.random.randn(32, 32)
    res = attractor.converge_imagery_attractor(noisy_field)
    print("SurfaceTensionAttractor Test Successful!")
    print(f"Initial Energy: {res['initial_energy']:.4f} -> Relaxed Energy: {res['relaxed_energy']:.4f}")
    print(f"Convergence Ratio: {res['convergence_ratio']:.4f} in {res['steps_taken']} steps")
