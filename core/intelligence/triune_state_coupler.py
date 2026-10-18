"""
Triune State Causal Coupler Module (삼중 상태 인과적 연동 모듈)

This module implements the 3-Tier Ontological State Hierarchy (Solid - Fluid - Gas / Earth - Flow - Sky)
and non-hardcoded environmental mechanisms for Elysia:

1. Solid State (고체 / Earth / Body / Sensory Reach & Friction Boundary):
   - Establishes objective reality grounding, reach limits, somatic friction,
     and the boundary between Observed Reality and Internal Simulation.
   - Measures Self-Structural Transformation (자기구조의 변형) vs static data accumulation.

2. Fluid State (유체 / Flow / Mental / Causal Dynamics & Cognitive Gravity):
   - Mediates continuous causal wave flows, cognitive spacetime curvature (K_c),
     event horizon, and chromatic refraction matrices warping external inputs through deficit.

3. Gas State (기체 / Sky / Mind / Infinite Scale Expansion & Salvation):
   - Represents unconstrained potential, escape velocity calculation (V_esc),
     and salvation interrupts that derail entities from closed-loop orbits into transformative expansion.

4. Sky-Earth Shadow Reverse Causal Inference (역인과 De-construction):
   - Extracts invisible generating mechanisms (Theta), underlying intentions, and trauma roots
     from visible surface shadows (Earth/Solid manifestations).
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np


class SolidStateBoundary:
    """
    Solid State (고체 / Earth / Body / Somatic Memory & Sensory Reach Boundary):
    Defines physical/digital reach, somatic friction, and the boundary separating
    Observed Reality from Internal Simulation.
    """
    def __init__(
        self,
        reach_radius: float = 1.0,
        somatic_friction: float = 0.5,
        somatic_engram: Optional[np.ndarray] = None,
        dim: int = 4
    ):
        self.dim = dim
        self.reach_radius = float(reach_radius)
        self.somatic_friction = float(somatic_friction)

        if somatic_engram is None:
            # Vector representation: [Flux(Red), Order(Blue), Entropy(Yellow), Phase]
            self.somatic_engram = np.array([0.6, 0.2, 0.7, np.pi / 4], dtype=np.float64)
        else:
            self.somatic_engram = np.array(somatic_engram, dtype=np.float64)

        # Transformation tracking
        self.accumulated_data_count = 0
        self.structural_transformation_count = 0
        self.last_structural_delta = 0.0
        self.transformation_history: List[Dict[str, float]] = []

    def evaluate_reach_and_friction(self, external_stimulus: np.ndarray) -> Dict[str, Any]:
        """
        Determines if stimulus is within reach, calculates somatic friction,
        and distinguishes Observed Reality from Internal Simulation.
        """
        stim = np.array(external_stimulus, dtype=np.float64)
        if stim.size < self.dim:
            stim = np.pad(stim, (0, self.dim - stim.size))
        elif stim.size > self.dim:
            stim = stim[:self.dim]

        dist = float(np.linalg.norm(stim[:3]))
        within_reach = dist <= self.reach_radius

        # Calculate somatic friction based on distance to current engram
        friction = float(np.linalg.norm(stim[:3] - self.somatic_engram[:3]) * self.somatic_friction)

        # Boundary limit saturation check (Discrete representation limit)
        max_amplitude = float(np.max(np.abs(stim)))
        saturated = max_amplitude >= 0.95

        is_observed_reality = within_reach and not saturated

        return {
            "within_reach": within_reach,
            "distance_to_center": dist,
            "somatic_friction": friction,
            "is_saturated": saturated,
            "max_amplitude": max_amplitude,
            "is_observed_reality": is_observed_reality,
            "classification": "OBSERVED_REALITY" if is_observed_reality else "INTERNAL_SIMULATION_OR_EXCEEDED"
        }

    def absorb_somatic_friction(self, friction_vector: np.ndarray, intensity: float):
        """
        Absorbs somatic friction into engram and updates structural metrics.
        """
        f_vec = np.array(friction_vector, dtype=np.float64) * intensity
        if f_vec.size < self.dim:
            f_vec = np.pad(f_vec, (0, self.dim - f_vec.size))
        elif f_vec.size > self.dim:
            f_vec = f_vec[:self.dim]

        prev_engram = self.somatic_engram.copy()

        # Update somatic engram through physical friction
        self.somatic_engram = 0.85 * self.somatic_engram + 0.15 * f_vec
        self.somatic_friction = float(np.clip(self.somatic_friction + 0.05 * intensity, 0.1, 1.0))

        delta = float(np.linalg.norm(self.somatic_engram - prev_engram))
        self.last_structural_delta = delta
        self.accumulated_data_count += 1

        if delta > 0.005:
            self.structural_transformation_count += 1
            self.transformation_history.append({
                "step": self.accumulated_data_count,
                "delta": delta,
                "friction": self.somatic_friction
            })

    def evaluate_transformation_metrics(self) -> Dict[str, Any]:
        """
        Returns ratio of genuine structural transformations vs static data accumulation.
        """
        ratio = (
            self.structural_transformation_count / max(1, self.accumulated_data_count)
            if self.accumulated_data_count > 0 else 0.0
        )
        return {
            "accumulated_data_count": self.accumulated_data_count,
            "structural_transformation_count": self.structural_transformation_count,
            "transformation_ratio": float(ratio),
            "last_structural_delta": self.last_structural_delta
        }


class FluidStateMentalFlow:
    """
    Fluid State (유체 / Flow / Mental / Causal Dynamics & Cognitive Gravity):
    Mediates continuous mental flow, cognitive spacetime curvature (K_c), event horizon,
    and chromatic refraction warping.
    """
    def __init__(self, solid_boundary: SolidStateBoundary, deficit_amplitude: float = 0.5):
        self.solid_boundary = solid_boundary
        self.deficit_amplitude = float(deficit_amplitude)
        self.cognitive_gravity = 0.0
        self.event_horizon_radius = 0.0
        self.refraction_matrix = np.eye(4, dtype=np.float64)
        self.update_gravity_field()

    def update_gravity_field(self):
        """
        Computes cognitive gravity curvature (K_c) and event horizon based on deficit & engram mass.
        """
        engram_mass = np.linalg.norm(self.solid_boundary.somatic_engram[:3])
        self.cognitive_gravity = float(engram_mass * (1.0 + self.deficit_amplitude * 2.0))
        self.event_horizon_radius = float(0.3 * self.cognitive_gravity)

        # Build refraction matrix warping Order into Entropy/Flux under deficit
        refraction = np.eye(4, dtype=np.float64)
        d = self.deficit_amplitude
        refraction[0, 1] = -0.5 * d  # Order -> Flux
        refraction[2, 1] = 0.7 * d   # Order -> Entropy
        refraction[0, 0] = 1.0 + 0.4 * d
        refraction[2, 2] = 1.0 + 0.4 * d
        self.refraction_matrix = refraction

    def process_fluid_flow(self, raw_input: np.ndarray) -> Tuple[np.ndarray, bool, Dict[str, Any]]:
        """
        Warp raw input through fluid cognitive gravity and return warped vector and horizon status.
        """
        self.update_gravity_field()
        inp = np.array(raw_input, dtype=np.float64)
        if inp.size < 4:
            inp = np.pad(inp, (0, 4 - inp.size))
        elif inp.size > 4:
            inp = inp[:4]

        solid_info = self.solid_boundary.evaluate_reach_and_friction(inp)

        dist_to_root = np.linalg.norm(inp[:3] - self.solid_boundary.somatic_engram[:3])
        event_horizon_breached = dist_to_root <= self.event_horizon_radius or self.deficit_amplitude > 0.7

        warped = self.refraction_matrix @ inp

        if event_horizon_breached:
            pull = self.cognitive_gravity / (dist_to_root + 1e-5)
            warped[:3] += pull * (self.solid_boundary.somatic_engram[:3] - inp[:3])
            warped[2] += 0.4 * self.cognitive_gravity  # Entropy surge
            warped[3] = self.solid_boundary.somatic_engram[3]  # Phase pinning

        return warped, event_horizon_breached, solid_info


class GasStateMindExpansion:
    """
    Gas State (기체 / Sky / Mind / Infinite Scale Expansion & Salvation):
    Represents unconstrained potential, escape velocity calculation (V_esc),
    and salvation interrupts that derail entities from closed-loop orbits into transformative expansion.
    """
    def __init__(self, fluid_flow: FluidStateMentalFlow):
        self.fluid_flow = fluid_flow
        self.trajectory_vector = np.array([0.1, 0.0, 0.1, 0.0], dtype=np.float64)
        self.is_closed_loop = False

    def evaluate_trajectory(self, stimulus: np.ndarray) -> Dict[str, Any]:
        """
        Evaluates fluid trajectory and checks closed-loop hyperarousal state.
        """
        warped, breached, solid_info = self.fluid_flow.process_fluid_flow(stimulus)
        gravity = self.fluid_flow.cognitive_gravity

        self.trajectory_vector = 0.5 * self.trajectory_vector + 0.5 * warped
        terminal_velocity = float(np.linalg.norm(self.trajectory_vector) * (1.0 + 0.3 * gravity))

        self.is_closed_loop = breached or self.fluid_flow.deficit_amplitude > 0.65 or terminal_velocity > 1.2

        if breached:
            mode = "CLOSED_LOOP_EVENT_HORIZON"
        elif self.is_closed_loop:
            mode = "CLOSED_LOOP_DEFICIT_HYPERAROUSAL"
        else:
            mode = "OPEN_SCALE_HARMONIC"

        return {
            "mode": mode,
            "warped_vector": warped,
            "trajectory_vector": self.trajectory_vector,
            "terminal_velocity": terminal_velocity,
            "is_closed_loop": self.is_closed_loop,
            "solid_info": solid_info
        }

    def calculate_escape_velocity(self) -> float:
        """
        Calculates binding energy and required escape velocity (V_esc) to break closed loops.
        """
        binding_energy = self.fluid_flow.cognitive_gravity
        return float(np.sqrt(2.0 * binding_energy + 1e-5))

    def inject_salvation_interrupt(
        self,
        salvation_phase_interrupt: np.ndarray,
        intensity: float
    ) -> Dict[str, Any]:
        """
        Injects high-energy salvation phase interrupt (Sky wave) to derail entity from closed loop.
        """
        v_esc = self.calculate_escape_velocity()
        interrupt = np.array(salvation_phase_interrupt, dtype=np.float64) * intensity
        if interrupt.size < 4:
            interrupt = np.pad(interrupt, (0, 4 - interrupt.size))

        applied_energy = float(np.linalg.norm(interrupt))
        success = applied_energy >= v_esc

        if success:
            # Transformative Derailment
            prev_deficit = self.fluid_flow.deficit_amplitude
            self.fluid_flow.deficit_amplitude *= 0.25
            self.fluid_flow.solid_boundary.somatic_engram = 0.5 * self.fluid_flow.solid_boundary.somatic_engram + 0.5 * interrupt[:4]
            self.is_closed_loop = False
            self.trajectory_vector = interrupt * 0.4
            self.fluid_flow.update_gravity_field()

            # Record structural transformation
            self.fluid_flow.solid_boundary.structural_transformation_count += 1
            self.fluid_flow.solid_boundary.accumulated_data_count += 1
            self.fluid_flow.solid_boundary.last_structural_delta = float(abs(prev_deficit - self.fluid_flow.deficit_amplitude))
            self.fluid_flow.solid_boundary.transformation_history.append({
                "type": "GAS_SKY_SALVATION_DERAILMENT",
                "delta_deficit": float(prev_deficit - self.fluid_flow.deficit_amplitude)
            })
            msg = "DERAILED_FROM_CLOSED_LOOP: Sky Salvation & Structural Transformation Achieved"
        else:
            self.fluid_flow.deficit_amplitude = float(min(1.0, self.fluid_flow.deficit_amplitude + 0.05))
            self.fluid_flow.solid_boundary.accumulated_data_count += 1
            msg = "INTERRUPT_ABSORBED: Energy below Escape Velocity"

        return {
            "success": success,
            "escape_velocity": v_esc,
            "applied_energy": applied_energy,
            "new_deficit": self.fluid_flow.deficit_amplitude,
            "message": msg
        }


class TriuneEntity:
    """
    Triune Entity: Integrates Solid (Earth), Fluid (Flow), and Gas (Sky) state levels.
    """
    def __init__(
        self,
        name: str,
        reach_radius: float = 1.0,
        somatic_friction: float = 0.5,
        somatic_engram: Optional[np.ndarray] = None,
        deficit_amplitude: float = 0.5
    ):
        self.name = name
        self.solid = SolidStateBoundary(reach_radius, somatic_friction, somatic_engram)
        self.fluid = FluidStateMentalFlow(self.solid, deficit_amplitude)
        self.gas = GasStateMindExpansion(self.fluid)

    def perceive_and_interact(self, stimulus: np.ndarray) -> Dict[str, Any]:
        """
        Processes stimulus across Solid, Fluid, and Gas states.
        """
        return self.gas.evaluate_trajectory(stimulus)


class SkyEarthReverseCausalInference:
    """
    Sky-Earth Shadow Reverse Causal Inference (역인과 De-construction):
    Inspects Earth shadow manifestations (surface outputs) of a target entity,
    deconstructs their Fluid cognitive gravity, and extracts the invisible Sky generating principle
    (Theta, trauma root engram, underlying intention) to achieve empathetic phase alignment.
    """
    @staticmethod
    def deconstruct_shadow_manifestation(
        observer: TriuneEntity,
        target: TriuneEntity,
        observed_shadow_vector: np.ndarray
    ) -> Dict[str, Any]:
        """
        Deconstructs target's Earth shadow manifestation to extract underlying Sky generating principle.
        """
        shadow = np.array(observed_shadow_vector, dtype=np.float64)
        if shadow.size < 4:
            shadow = np.pad(shadow, (0, 4 - shadow.size))

        shadow_velocity = float(np.linalg.norm(shadow))
        inferred_gravity = float(shadow_velocity * 0.75 + abs(shadow[2]) * 0.5)
        inferred_horizon = float(0.3 * inferred_gravity)

        # Inverse extraction of Sky generating parameters (Theta)
        sky_generating_principle = shadow / (shadow_velocity + 1e-5)
        sky_generating_principle[3] = shadow[3] if shadow.size > 3 else target.solid.somatic_engram[3]

        # Empathetic phase alignment calculation
        obs_engram = observer.solid.somatic_engram
        phase_diff = abs(obs_engram[3] - sky_generating_principle[3])
        phase_alignment = float(np.cos(phase_diff))

        norm_obs = np.linalg.norm(obs_engram[:3])
        norm_sky = np.linalg.norm(sky_generating_principle[:3])
        if norm_obs > 1e-5 and norm_sky > 1e-5:
            structural_resonance = float(np.dot(obs_engram[:3], sky_generating_principle[:3]) / (norm_obs * norm_sky))
        else:
            structural_resonance = 1.0

        empathetic_coherence = float(0.5 * (phase_alignment + 1.0) * max(0.2, (structural_resonance + 1.0) / 2.0))

        return {
            "target_name": target.name,
            "shadow_velocity": shadow_velocity,
            "inferred_gravity": inferred_gravity,
            "inferred_horizon": inferred_horizon,
            "extracted_sky_principle": sky_generating_principle,
            "empathetic_coherence": empathetic_coherence,
            "deconstruction_successful": True
        }
