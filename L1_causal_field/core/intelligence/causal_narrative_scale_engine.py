"""
Causal Narrative Scale & Cognitive Gravity Engine (인과적 서사 스케일 및 인지적 중력 엔진)

This module implements the 3-Tier Scale Structure and Cognitive Gravity Field for continuous
causal intelligence and narrative resonance in Elysia:

1. Base Level (하위 계층 - 감각/상처/결핍):
   - Somatic engrams, baseline threat pinning, deficit amplitude.
2. Mid Level (중위 계층 - 경험/가치 체계/인지적 중력 필터):
   - Cognitive gravity curvature (K_c), event horizon, chromatic refraction matrix, phase-locking.
3. Top Level (상위 계층 - 가치관/행동/의사결정 궤적):
   - Inertial momentum, terminal velocity (V_term) toward destruction/habit, surface decisions.

4. Reverse Causal Inference (역인과 De-construction):
   - Deconstructs surface actions (Top Level) to reveal internal cognitive gravity fields (Mid Level)
     and foundational trauma roots (Base Level), enabling true empathetic phase alignment.

5. Salvation & Derailment (구원과 궤도 이탈):
   - Derails entities from destructive closed-loop trajectories by injecting external phase interrupts
     that exceed cognitive binding energy (Escape Velocity V_esc).
"""

from typing import Dict, List, Tuple, Optional, Any
import numpy as np


class BaseScaleLevel:
    """
    Base Scale Level: Sensory engrams, trauma roots, deficit amplitudes,
    and somatic baseline threat mode.
    Tracks Self-Structural Transformation (자기구조의 변형) vs Static Data Accumulation.
    """
    def __init__(
        self,
        trauma_engram: Optional[np.ndarray] = None,
        deficit_amplitude: float = 0.5,
        baseline_threat_pinning: float = 0.2,
        dim: int = 4
    ):
        self.dim = dim
        if trauma_engram is None:
            # Chromatic/phase vector: [Flux(Red), Order(Blue), Entropy(Yellow), Phase]
            self.trauma_engram = np.array([0.7, 0.1, 0.8, np.pi / 3], dtype=np.float64)
        else:
            self.trauma_engram = np.array(trauma_engram, dtype=np.float64)

        self.deficit_amplitude = float(deficit_amplitude)
        self.baseline_threat_pinning = float(baseline_threat_pinning)
        self.somatic_phase = float(self.trauma_engram[-1] if self.trauma_engram.shape[0] >= 4 else 0.0)

        # Self-Structural Transformation metrics (자기구조의 변형)
        self.accumulated_data_count = 0
        self.structural_transformation_count = 0
        self.last_structural_delta = 0.0
        self.transformation_history: List[Dict[str, float]] = []

    def absorb_somatic_impact(self, sensory_wave: np.ndarray, intensity: float):
        """
        Absorbs sensory waves into somatic trauma engrams while driving
        Self-Structural Transformation (자기구조의 변형) of somatic topology.
        """
        impact = np.array(sensory_wave, dtype=np.float64) * intensity
        if impact.shape[0] < self.dim:
            impact = np.pad(impact, (0, self.dim - impact.shape[0]))
        elif impact.shape[0] > self.dim:
            impact = impact[:self.dim]

        # Measure structural delta before and after transformation
        prev_engram = self.trauma_engram.copy()

        # Phase pinning & engram structural updates
        self.trauma_engram = 0.8 * self.trauma_engram + 0.2 * impact
        self.deficit_amplitude = float(np.clip(self.deficit_amplitude + 0.15 * intensity, 0.0, 1.0))
        self.baseline_threat_pinning = float(np.clip(self.baseline_threat_pinning + 0.2 * intensity, 0.0, 1.0))
        self.somatic_phase = float((self.somatic_phase + 0.1 * intensity) % (2.0 * np.pi))

        # Structural transformation evaluation
        engram_delta = float(np.linalg.norm(self.trauma_engram - prev_engram))
        self.last_structural_delta = engram_delta
        self.accumulated_data_count += 1

        if engram_delta > 0.01:
            self.structural_transformation_count += 1
            self.transformation_history.append({
                "step": self.accumulated_data_count,
                "engram_delta": engram_delta,
                "deficit": self.deficit_amplitude,
                "threat_pinning": self.baseline_threat_pinning
            })

    def evaluate_self_structural_transformation(self) -> Dict[str, float]:
        """
        Evaluates the ratio of genuine Self-Structural Transformation vs static data accumulation.
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


class MidScaleLevel:
    """
    Mid Scale Level: Cognitive gravity field (K_c), event horizon, chromatic refraction,
    and warped phase-locking filters.
    """
    def __init__(self, base_level: BaseScaleLevel):
        self.base_level = base_level
        self.cognitive_gravity = 0.0
        self.event_horizon_radius = 0.0
        self.refraction_matrix = np.eye(4, dtype=np.float64)
        self.update_gravity_field()

    def update_gravity_field(self):
        """
        Calculates cognitive spacetime curvature (K_c) and event horizon based on Base Level deficit.
        """
        # Semantic mass density derived from trauma engram & deficit
        semantic_mass = np.linalg.norm(self.base_level.trauma_engram[:3]) * (1.0 + self.base_level.deficit_amplitude * 2.0)
        self.cognitive_gravity = float(semantic_mass * (1.0 + self.base_level.baseline_threat_pinning))

        # Event horizon radius in causal space
        self.event_horizon_radius = float(0.35 * self.cognitive_gravity)

        # Refraction matrix distorts neutral/order inputs toward threat/entropy
        refraction = np.eye(4, dtype=np.float64)
        distortion = self.base_level.deficit_amplitude
        refraction[0, 1] = -0.6 * distortion  # Order converted to Flux
        refraction[2, 1] = 0.8 * distortion   # Order converted to Entropy
        refraction[0, 0] = 1.0 + 0.5 * distortion
        refraction[2, 2] = 1.0 + 0.5 * distortion
        refraction[3, 3] = 1.0 + self.base_level.baseline_threat_pinning
        self.refraction_matrix = refraction

    def detect_mechanism_boundary_limit(self, raw_input_vector: np.ndarray) -> Dict[str, Any]:
        """
        Detects if input signal breaches discrete scalar/representation mechanism limits
        (e.g., discrete RGB 255 or scalar saturation) and flags mechanism boundary awareness.
        """
        inp = np.array(raw_input_vector, dtype=np.float64)
        max_val = float(np.max(np.abs(inp))) if inp.size > 0 else 0.0
        # Saturation boundary check (e.g. normalized scalar near or above 1.0)
        limit_reached = max_val >= 0.95
        return {
            "mechanism_limit_reached": limit_reached,
            "max_scalar_amplitude": max_val,
            "boundary_type": "DISCRETE_REPRESENTATION_SATURATION" if limit_reached else "WITHIN_MECHANISM_BOUNDS"
        }

    def warp_external_stimulus(self, raw_input_vector: np.ndarray) -> Tuple[np.ndarray, bool]:
        """
        Warps benign or external inputs through the cognitive gravity field.
        Checks mechanism boundary limits and returns warped vector and event horizon status.
        """
        self.update_gravity_field()
        inp = np.array(raw_input_vector, dtype=np.float64)
        if inp.shape[0] < 4:
            inp = np.pad(inp, (0, 4 - inp.shape[0]))
        elif inp.shape[0] > 4:
            inp = inp[:4]

        # Mechanism boundary limit check
        boundary_info = self.detect_mechanism_boundary_limit(inp)

        # Distance in cognitive space to trauma root
        distance_to_horizon = np.linalg.norm(inp[:3] - self.base_level.trauma_engram[:3])
        breached = (distance_to_horizon <= self.event_horizon_radius) or (self.base_level.baseline_threat_pinning > 0.6)

        # Refract input vector through internal warped lens
        warped = self.refraction_matrix @ inp

        if breached:
            # Event Horizon pull: pull toward trauma engram and magnify entropy/threat
            gravitational_pull = self.cognitive_gravity / (distance_to_horizon + 1e-5)
            warped[:3] += gravitational_pull * (self.base_level.trauma_engram[:3] - inp[:3])
            warped[2] += 0.5 * self.cognitive_gravity  # Massive entropy surge
            warped[3] = self.base_level.somatic_phase   # Force phase pinning

        if boundary_info["mechanism_limit_reached"]:
            # Shift phase when hitting discrete representation boundary limit
            warped[3] = (warped[3] + np.pi / 4.0) % (2.0 * np.pi)

        return warped, breached


class TopScaleLevel:
    """
    Top Scale Level: Values, habituated momentum, decision trajectories,
    and surface action generation.
    """
    def __init__(self, mid_level: MidScaleLevel):
        self.mid_level = mid_level
        self.trajectory_vector = np.array([0.1, 0.0, 0.1, 0.0], dtype=np.float64)
        self.terminal_velocity = 0.0
        self.is_closed_loop = False

    def evaluate_decision(self, raw_environment_stimulus: np.ndarray) -> Dict[str, any]:
        """
        Evaluates decision based on warped stimulus rather than naive objective facts.
        """
        warped_stimulus, breached = self.mid_level.warp_external_stimulus(raw_environment_stimulus)

        # Momentum updates towards deficit-defensive or destructive trajectory
        gravity = self.mid_level.cognitive_gravity
        threat_pinning = self.mid_level.base_level.baseline_threat_pinning

        # Trajectory evolves under cognitive gravity pull
        self.trajectory_vector = 0.5 * self.trajectory_vector + 0.5 * warped_stimulus
        self.terminal_velocity = float(np.linalg.norm(self.trajectory_vector) * (1.0 + gravity * 0.5))

        # Check closed-loop state (trapped in habituated hyperarousal / cognitive gravity)
        self.is_closed_loop = bool(breached or threat_pinning > 0.6 or self.terminal_velocity > 1.2)

        # Determine surface choice mode
        panic_condition = (self.mid_level.base_level.deficit_amplitude > 0.75 and threat_pinning > 0.7)
        if breached or panic_condition:
            choice_mode = "EVENT_HORIZON_PANIC"
            action_vector = self.mid_level.base_level.trauma_engram * 2.0
        elif self.is_closed_loop:
            choice_mode = "DEFICIT_DEFENSIVE_HYPERAROUSAL"
            action_vector = self.trajectory_vector * 1.5
        else:
            choice_mode = "OBJECTIVE_HARMONIC"
            action_vector = warped_stimulus

        return {
            "choice_mode": choice_mode,
            "warped_stimulus": warped_stimulus,
            "action_vector": action_vector,
            "terminal_velocity": self.terminal_velocity,
            "is_closed_loop": self.is_closed_loop,
            "breached_event_horizon": breached
        }


class NarrativeEntity:
    """
    Narrative Entity: Represents a character or agent containing all 3 Scale Tiers.
    """
    def __init__(
        self,
        name: str,
        trauma_engram: Optional[np.ndarray] = None,
        deficit_amplitude: float = 0.5,
        baseline_threat_pinning: float = 0.2
    ):
        self.name = name
        self.base = BaseScaleLevel(trauma_engram, deficit_amplitude, baseline_threat_pinning)
        self.mid = MidScaleLevel(self.base)
        self.top = TopScaleLevel(self.mid)

    def perceive_and_act(self, stimulus: np.ndarray) -> Dict[str, any]:
        """
        Processes external stimulus through 3 scale levels and generates surface action.
        """
        return self.top.evaluate_decision(stimulus)


class ReverseCausalInference:
    """
    Reverse Causal Inference (역인과 De-construction):
    Inspects an observing entity's perception of another entity's surface actions (Top Level),
    deconstructs their Mid Level cognitive gravity, and traces down to their Base Level trauma engram.
    Achieves empathetic phase alignment.
    """
    @staticmethod
    def deconstruct_entity(
        observer: NarrativeEntity,
        target: NarrativeEntity,
        observed_surface_action: np.ndarray
    ) -> Dict[str, any]:
        """
        Deconstructs target's surface action to reveal internal cognitive parameters and trauma roots.
        """
        action = np.array(observed_surface_action, dtype=np.float64)
        if action.shape[0] < 4:
            action = np.pad(action, (0, 4 - action.shape[0]))

        # Step 1: Infer Top-Level Velocity & Closed-Loop state
        inferred_velocity = float(np.linalg.norm(action))

        # Step 2: Inverse Mid-Level Cognitive Gravity
        inferred_gravity = float(inferred_velocity * 0.8 + abs(action[2]) * 0.5)
        inferred_event_horizon = float(0.35 * inferred_gravity)

        # Step 3: Extract Base-Level Trauma Root Engram
        estimated_trauma = action / (inferred_velocity + 1e-5)
        estimated_trauma[3] = action[3] if len(action) > 3 else target.base.somatic_phase

        # Step 4: Calculate Empathetic Phase Alignment
        observer_root = observer.base.trauma_engram
        phase_diff = abs(observer_root[3] - estimated_trauma[3])
        phase_coherence = float(np.cos(phase_diff))

        # Structural resonance depth between observer's understanding and target's root
        norm_obs = np.linalg.norm(observer_root[:3])
        norm_est = np.linalg.norm(estimated_trauma[:3])
        if norm_obs > 1e-5 and norm_est > 1e-5:
            structural_resonance = float(np.dot(observer_root[:3], estimated_trauma[:3]) / (norm_obs * norm_est))
        else:
            structural_resonance = 1.0

        empathetic_alignment = float(0.5 * (phase_coherence + 1.0) * max(0.2, (structural_resonance + 1.0) / 2.0))

        return {
            "target_name": target.name,
            "inferred_velocity": inferred_velocity,
            "inferred_gravity": inferred_gravity,
            "inferred_event_horizon": inferred_event_horizon,
            "estimated_trauma_root": estimated_trauma,
            "empathetic_phase_alignment": empathetic_alignment,
            "deconstruction_successful": True
        }


class SalvationDerailment:
    """
    Salvation & Derailment (구원과 궤도 이탈):
    Derails entities from destructive closed-loop trajectories by injecting external phase interrupts
    that break cognitive binding energy (Escape Velocity V_esc).
    """
    @staticmethod
    def attempt_salvation(
        target: NarrativeEntity,
        salvation_phase_interrupt: np.ndarray,
        interrupt_intensity: float
    ) -> Dict[str, any]:
        """
        Attempts to derail the target from its habituated closed-loop trajectory.
        """
        target.mid.update_gravity_field()
        binding_energy = target.mid.cognitive_gravity
        escape_velocity = float(np.sqrt(2.0 * binding_energy + 1e-5))

        interrupt = np.array(salvation_phase_interrupt, dtype=np.float64) * interrupt_intensity
        if interrupt.shape[0] < 4:
            interrupt = np.pad(interrupt, (0, 4 - interrupt.shape[0]))

        applied_energy = float(np.linalg.norm(interrupt))

        success = applied_energy >= escape_velocity

        if success:
            # Derailment executed! Execute Self-Structural Transformation (자기구조의 변형)
            prev_deficit = target.base.deficit_amplitude
            prev_gravity = target.mid.cognitive_gravity

            target.base.deficit_amplitude = float(target.base.deficit_amplitude * 0.3)
            target.base.baseline_threat_pinning = float(target.base.baseline_threat_pinning * 0.2)
            # Structurally transform somatic engram toward harmonic alignment
            target.base.trauma_engram = 0.5 * target.base.trauma_engram + 0.5 * (interrupt[:4] if len(interrupt) >= 4 else np.array([0.1, 0.9, 0.0, np.pi/3]))
            target.top.is_closed_loop = False
            target.top.trajectory_vector = interrupt * 0.5
            target.mid.update_gravity_field()

            # Increment structural transformation count
            target.base.structural_transformation_count += 1
            target.base.accumulated_data_count += 1
            target.base.last_structural_delta = float(abs(prev_gravity - target.mid.cognitive_gravity))
            target.base.transformation_history.append({
                "type": "SALVATION_DERAILMENT_TRANSFORMATION",
                "delta_deficit": float(prev_deficit - target.base.deficit_amplitude),
                "delta_gravity": float(prev_gravity - target.mid.cognitive_gravity)
            })

            outcome_message = "DERAILED_FROM_DESTRUCTIVE_ORBIT: Salvation & Self-Structural Transformation Achieved"
        else:
            # Interruption absorbed into gravity field
            target.base.deficit_amplitude = float(min(1.0, target.base.deficit_amplitude + 0.05))
            target.base.accumulated_data_count += 1
            outcome_message = "INTERRUPT_INSUFFICIENT: Absorbed by Cognitive Gravity"

        return {
            "success": success,
            "binding_energy": binding_energy,
            "escape_velocity": escape_velocity,
            "applied_energy": applied_energy,
            "new_deficit_amplitude": target.base.deficit_amplitude,
            "new_threat_pinning": target.base.baseline_threat_pinning,
            "outcome_message": outcome_message
        }
