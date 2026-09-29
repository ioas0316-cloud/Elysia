"""
Elysia Ontological Fluid Medium Engine
======================================
Core module realizing the Causal Principle Medium (원리가 매질이 되는 인과 엔진):
1. Topological Boundary Field (위상적 경계 공간): Flexible topological space defining internal state vs external resistance.
2. Relational Coupling Rules (관계의 규칙): Density, cohesion, phase viscosity, chromatic flux, and causal impedance.
3. Ignorance Gradient & Tension (무지의 구배 & 갈망): Measuring phase divergence between internal expectation and reality.
4. Emergent Phenomena (자연스러운 발현): Wave propagation, reflection, refraction, and phase transition (water <-> ice)
   emerging from structural medium dynamics without hardcoded optical/ray-bounce formulas.
5. Principle Invariant Extraction (원리적 동일성 추출): Observing and extracting underlying causal invariants directly from medium dynamics.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from core.consciousness.triadic_boundary_causal_engine import TriadicBoundaryCausalEngine


class OntologicalFluidMedium:
    """
    [OntologicalFluidMedium - 존재론적 파동 매질]
    Replaces brute-force numerical ray-tracing with structural medium rules.
    Refraction, reflection, ripples, and phase cohesion naturally emerge from
    causal impedance gradients (Z), phase viscosity (eta), and density (rho).
    """
    def __init__(
        self,
        grid_shape: Tuple[int, int] = (32, 32),
        base_density: float = 1.0,
        base_viscosity: float = 0.8,
        triadic_engine: Optional[TriadicBoundaryCausalEngine] = None
    ):
        self.rows, self.cols = grid_shape
        self.shape = grid_shape

        # 1. Physical-Causal Medium Field
        # Density field rho(x, y)
        self.density = np.full(grid_shape, base_density, dtype=np.float32)
        # Velocity field (Vx, Vy)
        self.velocity_x = np.zeros(grid_shape, dtype=np.float32)
        self.velocity_y = np.zeros(grid_shape, dtype=np.float32)
        # Pressure / Potential field P(x, y)
        self.pressure = np.zeros(grid_shape, dtype=np.float32)
        # Wave amplitude / Displacement field phi(x, y)
        self.wave_field = np.zeros(grid_shape, dtype=np.float32)
        # Phase Angle theta(x, y) in [0, 2pi)
        self.phase_angle = np.zeros(grid_shape, dtype=np.float32)

        # 2. Structural & Material Attributes
        # Phase Viscosity / Topological Cohesion (eta)
        self.viscosity = np.full(grid_shape, base_viscosity, dtype=np.float32)
        # Causal Impedance Z(x, y) = sqrt(rho / eta)
        self.impedance = np.sqrt(self.density / (self.viscosity + 1e-6)).astype(np.float32)
        # Phase State: 0 = Fluid Water, 1 = Cohesive Water/Ice, -1 = Dispersed Vapor
        self.phase_state = np.zeros(grid_shape, dtype=np.int32)

        # 3. Chromatic Field: [Red (Flux), Blue (Order/Cohesion), Yellow (Entropy/Tension)]
        self.chromatic_field = np.zeros((*grid_shape, 3), dtype=np.float32)
        self.chromatic_field[:, :, 0] = 0.33  # Red
        self.chromatic_field[:, :, 1] = 0.50  # Blue
        self.chromatic_field[:, :, 2] = 0.17  # Yellow

        # 4. Ignorance Gradient & Triadic Boundary Integration
        self.triadic_engine = triadic_engine or TriadicBoundaryCausalEngine(dimension=16)
        self.tension_field = np.zeros(grid_shape, dtype=np.float32)
        self.yearning_gradient = np.zeros((*grid_shape, 2), dtype=np.float32)

        # Extracted Principle Schema (원리적 동일성 노드)
        self.extracted_principles: List[Dict[str, Any]] = []

    def set_medium_region(
        self,
        row_slice: slice,
        col_slice: slice,
        density: float,
        viscosity: float,
        phase_state: int = 0
    ):
        """Sets medium properties for a specific region (e.g. denser water body or boundary)."""
        self.density[row_slice, col_slice] = density
        self.viscosity[row_slice, col_slice] = viscosity
        self.phase_state[row_slice, col_slice] = phase_state
        # Recalculate emergent impedance Z = sqrt(rho / eta)
        self.impedance[row_slice, col_slice] = np.sqrt(density / (viscosity + 1e-6))

    def inject_impulse(
        self,
        pos: Tuple[int, int],
        amplitude: float = 2.0,
        flux_color: Tuple[float, float, float] = (0.8, 0.2, 0.1)
    ):
        """Injects a physical impulse / wave disturbance into the medium at a given position."""
        r, c = pos
        if 0 <= r < self.rows and 0 <= c < self.cols:
            self.wave_field[r, c] += amplitude
            self.pressure[r, c] += amplitude * 1.5
            self.chromatic_field[r, c] = np.array(flux_color, dtype=np.float32)

    def calculate_emergent_refractive_index(self) -> np.ndarray:
        """
        [Emergent Refractive Index n(x, y)]
        Refractive index is not hardcoded from Snell's law tables!
        It naturally emerges as the ratio of phase velocity in vacuum/air vs medium:
        n = c_0 / v_phase where v_phase = 1 / sqrt(rho * Z).
        """
        phase_velocity = 1.0 / (np.sqrt(self.density * self.impedance) + 1e-6)
        c_vacuum = 1.0  # Normalized reference phase velocity
        emergent_n = c_vacuum / (phase_velocity + 1e-6)
        return emergent_n.astype(np.float32)

    def compute_ignorance_gradient(self):
        """
        [Ignorance Gradient & Tension Computation (무지의 구배 측정)]
        Calculates local tension caused by phase mismatch between internal expectation
        and reality signal at boundary, creating a directional yearning gradient.
        """
        # Internal expectation field from triadic engine
        simulated_signal = self.triadic_engine.internal_world.simulate_expectation("visual")
        expectation_val = float(np.mean(simulated_signal[:4]))

        # Tension = |wave_field - expectation| * (1 + Yellow entropy)
        entropy = self.chromatic_field[:, :, 2]
        self.tension_field = np.abs(self.wave_field - expectation_val) * (1.0 + entropy)

        # Yearning gradient = -grad(Tension field)
        gy, gx = np.gradient(self.tension_field)
        self.yearning_gradient[:, :, 0] = -gy
        self.yearning_gradient[:, :, 1] = -gx

    def step(self, dt: float = 0.1):
        """
        Advances the ontological medium through continuous causal dynamics.
        Wave propagation, reflection, refraction, and phase cohesion happen
        organically without ray bouncing equations or if-else optical rules.
        """
        # 1. Compute pressure gradient forces
        dp_dy, dp_dx = np.gradient(self.pressure)

        # Acceleration a = -grad(P) / rho
        ax = -dp_dx / (self.density + 1e-6)
        ay = -dp_dy / (self.density + 1e-6)

        # 2. Update velocity field considering phase viscosity & yearning drive
        self.velocity_x = (self.velocity_x + ax * dt) * (1.0 - self.viscosity * dt * 0.1)
        self.velocity_y = (self.velocity_y + ay * dt) * (1.0 - self.viscosity * dt * 0.1)

        # Add yearning gradient impulse to velocity (drive toward tension relaxation)
        self.velocity_x += self.yearning_gradient[:, :, 1] * 0.05 * dt
        self.velocity_y += self.yearning_gradient[:, :, 0] * 0.05 * dt

        # 3. Wave field propagation via continuum wave equation
        # Laplacian of wave field d2(phi)
        d2y, d2x = np.gradient(np.gradient(self.wave_field)[0])[0], np.gradient(np.gradient(self.wave_field)[1])[1]
        laplacian_wave = d2x + d2y

        # Local wave phase speed c_phase^2 = 1 / (density * impedance)
        c_squared = 1.0 / (self.density * self.impedance + 1e-6)

        # Update pressure and wave field
        self.pressure += laplacian_wave * c_squared * dt
        self.pressure *= (1.0 - 0.02 * dt)  # Damping

        self.wave_field += (self.velocity_x + self.velocity_y + self.pressure) * dt
        self.wave_field *= 0.98  # Wave attenuation

        # 4. Phase Angle & Chromatic Coupling
        self.phase_angle = (self.phase_angle + np.hypot(self.velocity_x, self.velocity_y) * dt) % (2.0 * np.pi)

        # Chromatic transformation: High wave kinetic energy increases Flux (Red),
        # high cohesion increases Order (Blue), unabsorbed tension converts to Yellow (Entropy)
        kinetic_energy = 0.5 * self.density * (self.velocity_x**2 + self.velocity_y**2)
        self.chromatic_field[:, :, 0] = np.clip(self.chromatic_field[:, :, 0] + kinetic_energy * 0.1, 0.0, 1.0)
        self.chromatic_field[:, :, 1] = np.clip(self.viscosity * 0.8, 0.0, 1.0)
        self.chromatic_field[:, :, 2] = np.clip(self.tension_field * 0.2, 0.0, 1.0)

        # 5. Emergent Phase Transition (e.g., Water <-> Ice / Cohesive state)
        # When Order (Blue) > 0.75 and kinetic energy < 0.05, medium crystallizes into cohesive state (Ice)
        ice_mask = (self.chromatic_field[:, :, 1] > 0.75) & (kinetic_energy < 0.05)
        self.phase_state[ice_mask] = 1  # Ice / Solid state

        # 6. Re-evaluate ignorance gradient
        self.compute_ignorance_gradient()

    def extract_principle_isomorphism(self) -> Dict[str, Any]:
        """
        [Extract Principle Isomorphism - 원리적 동일성 추출]
        Extracts the underlying invariant principle (generating structure) directly from
        medium dynamics without fitting formulas or regression equations.
        Observes the relationship between Impedance Gradient and Wave Refraction Ratio.
        """
        emergent_n = self.calculate_emergent_refractive_index()

        # Invariant ratio = Mean(n_medium) / Mean(n_vacuum)
        mean_n = float(np.mean(emergent_n))
        std_n = float(np.std(emergent_n))
        density_contrast = float(np.max(self.density) / (np.min(self.density) + 1e-6))
        total_tension = float(np.sum(self.tension_field))

        principle_schema = {
            "principle_name": "EMERGENT_MEDIUM_REFRACTION_AND_PHASE_COHESION",
            "emergent_refractive_index_mean": mean_n,
            "emergent_refractive_index_std": std_n,
            "density_contrast_ratio": density_contrast,
            "total_ignorance_tension": total_tension,
            "phase_state_distribution": {
                "liquid_water": int(np.sum(self.phase_state == 0)),
                "cohesive_ice": int(np.sum(self.phase_state == 1)),
                "dispersed_vapor": int(np.sum(self.phase_state == -1))
            },
            "isomorphic_invariant": "PhaseVelocity_proportional_to_1_over_sqrt(rho*Z)"
        }

        self.extracted_principles.append(principle_schema)
        return principle_schema

    def get_state_summary(self) -> Dict[str, Any]:
        """Returns complete state summary of the ontological fluid medium."""
        return {
            "shape": self.shape,
            "avg_wave_amplitude": float(np.mean(np.abs(self.wave_field))),
            "max_wave_amplitude": float(np.max(np.abs(self.wave_field))),
            "avg_density": float(np.mean(self.density)),
            "avg_impedance": float(np.mean(self.impedance)),
            "emergent_refractive_index_mean": float(np.mean(self.calculate_emergent_refractive_index())),
            "total_tension": float(np.sum(self.tension_field)),
            "extracted_principles_count": len(self.extracted_principles)
        }
