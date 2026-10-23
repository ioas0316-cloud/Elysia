"""
Closed-Loop Valence Field & Sensory Feedback Engine for Elysia.

This module implements a closed-loop control logic that:
1. Computes prediction errors between internal predictions (Internal Forward Model)
   and external sensory streams S(t).
2. Performs real-time gradient descent feedback on ConceptAttractor spatial positions [x, y, z]
   and potential well depths (causal confidence/mass).
3. Evaluates dynamical phase trajectories (position x, velocity dx/dt) and supports
   dynamic latent dimension expansion (Latent Axis Spawning) when persistent,
   unexplainable prediction errors trigger an awareness of unobserved physical variables.
"""

from dataclasses import dataclass, field
import numpy as np
from typing import Dict, List, Optional, Tuple, Union


@dataclass
class ConceptAttractor:
    """Concept Attractor Well on the semantic valence potential field."""
    concept_id: str
    center_pos: np.ndarray       # Geometric position [x, y, z] in internal thought space
    resonant_freq: float        # Internal resonant frequency (Hz)
    well_depth: float           # Potential well depth (mass/valence weight)
    well_radius: float          # Potential attraction radius
    phase_velocity: Optional[np.ndarray] = None  # Rate-of-change trajectory vector dx/dt
    latent_dimensions: List[str] = field(default_factory=list)  # Dynamically spawned latent variable axes


class SensoryFeedbackEngine:
    """
    Sensing -> Prediction Error Computation -> Internal Potential Field State Adaptation (Closed-loop).
    """

    def __init__(
        self,
        spatial_learning_rate: float = 0.05,   # Spatial position inertia modification rate
        depth_adaptation_rate: float = 0.02,   # Well depth modification rate based on prediction accuracy
        error_decay: float = 0.95,             # Accumulated prediction error decay rate
        latent_spawn_threshold: float = 25.0   # Error threshold triggering latent dimension spawning
    ):
        self.lr_pos = spatial_learning_rate
        self.lr_depth = depth_adaptation_rate
        self.error_decay = error_decay
        self.latent_spawn_threshold = latent_spawn_threshold
        self.accumulated_errors: Dict[str, float] = {}
        self.spawned_latent_axes: Dict[str, int] = {}

    def predict_observation(self, attractor: ConceptAttractor) -> np.ndarray:
        """
        [1. Forward Path] Internal Model prediction of expected external sensory observations
        from current concept attractor state (positions, phase velocity, and resonant frequency).
        """
        # Base projection: center position (first 3 spatial coordinates) and normalized resonant frequency
        base_pos = attractor.center_pos[:3] if len(attractor.center_pos) >= 3 else attractor.center_pos
        base_projection = np.append(base_pos, attractor.resonant_freq / 100.0)

        # Include phase velocity trajectory if present
        phase_vel = getattr(attractor, 'phase_velocity', None)
        if phase_vel is not None:
            base_projection = np.concatenate([base_projection, phase_vel])

        # Include active latent dimension representations
        latent_dims = getattr(attractor, 'latent_dimensions', [])
        if latent_dims:
            latent_vector = np.array([0.1 * (idx + 1) for idx in range(len(latent_dims))])
            base_projection = np.concatenate([base_projection, latent_vector])

        return base_projection

    def calculate_prediction_error(
        self,
        predicted_sensory: np.ndarray,
        actual_sensory: np.ndarray
    ) -> Tuple[np.ndarray, float]:
        """
        [2. Sensing & Comparison] Calculates Surprise / Prediction Error between actual observation and internal model.
        """
        # Align lengths if vector dimensions differ
        if len(predicted_sensory) != len(actual_sensory):
            min_len = min(len(predicted_sensory), len(actual_sensory))
            error_vector = actual_sensory[:min_len] - predicted_sensory[:min_len]
        else:
            error_vector = actual_sensory - predicted_sensory

        error_magnitude = float(np.linalg.norm(error_vector))
        return error_vector, error_magnitude

    def check_and_spawn_latent_dimension(
        self,
        attractor: ConceptAttractor,
        accumulated_err: float
    ) -> bool:
        """
        Spawns a new latent variable dimension when persistent unexplainable prediction error
        exceeds the cognitive threshold, signifying an unobserved physical variable in the external world.
        """
        if accumulated_err > self.latent_spawn_threshold:
            spawn_count = self.spawned_latent_axes.get(attractor.concept_id, 0) + 1
            self.spawned_latent_axes[attractor.concept_id] = spawn_count
            axis_name = f"LATENT_AXIS_W{spawn_count}"
            if not hasattr(attractor, 'latent_dimensions'):
                attractor.latent_dimensions = []
            if axis_name not in attractor.latent_dimensions:
                attractor.latent_dimensions.append(axis_name)
                # Expand center_pos dimension to accommodate newly spawned latent axis
                attractor.center_pos = np.append(attractor.center_pos, 0.0)
                return True
        return False

    def adapt_attractor_state(
        self,
        attractor: ConceptAttractor,
        actual_sensory: np.ndarray
    ) -> Dict[str, Union[float, bool, List[str]]]:
        """
        [3. Closed-Loop Adaptation] Real-time adaptation of concept attractor position, phase velocity,
        and potential well depth based on sensory observation feedback.
        """
        predicted = self.predict_observation(attractor)
        error_vec, error_mag = self.calculate_prediction_error(predicted, actual_sensory)

        # Update accumulated prediction error (lower error indicates higher causal confidence)
        prev_err = self.accumulated_errors.get(attractor.concept_id, error_mag)
        updated_err = prev_err * self.error_decay + error_mag * (1 - self.error_decay)
        self.accumulated_errors[attractor.concept_id] = updated_err

        # A. Position (and Phase Velocity) Feedback Gradient
        pos_dim = len(attractor.center_pos)
        pos_error_gradient = error_vec[:pos_dim]
        if len(pos_error_gradient) < pos_dim:
            padded_gradient = np.zeros(pos_dim)
            padded_gradient[:len(pos_error_gradient)] = pos_error_gradient
            attractor.center_pos += self.lr_pos * padded_gradient
        else:
            attractor.center_pos += self.lr_pos * pos_error_gradient

        if len(error_vec) > pos_dim:
            vel_gradient = error_vec[pos_dim:pos_dim + (3 if len(error_vec) >= pos_dim + 3 else 1)]
            phase_vel = getattr(attractor, 'phase_velocity', None)
            if phase_vel is not None:
                attractor.phase_velocity += self.lr_pos * vel_gradient[:len(phase_vel)]
            else:
                attractor.phase_velocity = self.lr_pos * vel_gradient

        # B. Potential Well Depth Feedback
        # Accurate predictions deepen the well (reinforce causal confidence); large errors flatten it
        depth_delta = self.lr_depth * (1.0 / (1.0 + updated_err) - 0.5)
        attractor.well_depth = max(0.1, attractor.well_depth + depth_delta)

        # C. Unobserved Latent Axis Spawning Check
        spawned = self.check_and_spawn_latent_dimension(attractor, updated_err)

        active_dims = getattr(attractor, 'latent_dimensions', [])
        return {
            "prediction_error": error_mag,
            "accumulated_error": updated_err,
            "depth_delta": depth_delta,
            "latent_spawned": spawned,
            "active_latent_dims": list(active_dims)
        }


class ClosedLoopValenceField:
    """Valence Potential Field handling real-time sensorimotor friction and feedback loops."""

    def __init__(self):
        self.attractors: Dict[str, ConceptAttractor] = {}
        self.feedback_engine = SensoryFeedbackEngine()

    def register_attractor(self, attractor: ConceptAttractor):
        self.attractors[attractor.concept_id] = attractor

    def process_sensorimotor_step(self, sensor_stream: Dict[str, np.ndarray]) -> Dict[str, Dict]:
        """
        Receives real-time sensor stream data and updates all registered concept attractors.
        """
        telemetry = {}
        for concept_id, actual_sensory in sensor_stream.items():
            if concept_id in self.attractors:
                att = self.attractors[concept_id]
                metrics = self.feedback_engine.adapt_attractor_state(att, actual_sensory)
                telemetry[concept_id] = metrics
        return telemetry


# ==========================================
# Execution Simulation: Real-time Causal Adaptation via Observation Feedback
# ==========================================
if __name__ == "__main__":
    loop_field = ClosedLoopValenceField()

    # Initial 'Bulgogi Cooking State' Concept Attractor (Internal Prediction Model)
    # [Pan Temp (x), Maillard Reaction (y), Moisture Ratio (z)], Resonant Frequency (Hz)
    bulgogi_att = ConceptAttractor(
        concept_id="CONCEPT_COOKING_BULGOGI",
        center_pos=np.array([160.0, 0.2, 0.8]), # Initial prediction: 160C, Maillard 0.2, Moisture 0.8
        resonant_freq=120.0,
        well_depth=2.0,
        well_radius=1.5
    )
    loop_field.register_attractor(bulgogi_att)

    print("=== 0. Initial Concept Attractor State ===")
    print(f"Position: {bulgogi_att.center_pos.round(2)}, Well Depth (Causal Confidence): {bulgogi_att.well_depth:.3f}")

    # External Sensory Stream Simulation (Actual Sensor Measurements: pan hotter & drying faster than expected)
    # Sensor Input Vector: [Actual Temp, Actual Maillard, Actual Moisture, Frequency / 100]
    actual_sensory_stream = [
        np.array([175.0, 0.45, 0.6, 1.2]),  # Step 1: Hotter and drying faster than expected
        np.array([182.0, 0.65, 0.4, 1.2]),  # Step 2: High heat sustained
        np.array([185.0, 0.70, 0.3, 1.2]),  # Step 3: Convergence state
    ]

    print("\n=== Closed-Loop Sensory Feedback Loop Active ===")
    for step, sensor_data in enumerate(actual_sensory_stream, 1):
        telemetry = loop_field.process_sensorimotor_step({"CONCEPT_COOKING_BULGOGI": sensor_data})
        m = telemetry["CONCEPT_COOKING_BULGOGI"]

        print(f"\n[Step {step}] Sensory Feedback Received")
        print(f"  - Prediction Error (Surprise Magnitude): {m['prediction_error']:.3f}")
        print(f"  - Adjusted Attractor Position: {bulgogi_att.center_pos.round(2)}")
        print(f"  - Re-evaluated Well Depth: {bulgogi_att.well_depth:.3f} (delta: {m['depth_delta']:+.4f})")
