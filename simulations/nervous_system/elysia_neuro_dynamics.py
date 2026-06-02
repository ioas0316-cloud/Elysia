import numpy as np
import math
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os
import sys

# Ensure imports from parent directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulations.elysia_cognitive_spine import ElysiaCognitiveSpine, PhaseContrastGeneralizer

# Configure Korean Fonts
font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rc('font', family='NanumGothic')
    plt.rcParams['axes.unicode_minus'] = False

class HormonalFlux:
    """
    Manages the survival and differentiation voltages.
    Modulates input sensitivity (Weight) and Phase Mirror thresholds.
    """
    def __init__(self):
        # Baseline Hormones
        self.cortisol = 0.5   # Stress / Survival Alert (Modulates Input Weight)
        self.oxytocin = 0.5   # Resonance / Safety (Modulates Tolerance/Threshold)
        self.dopamine = 0.5   # Reward / Focus

    def set_state(self, cortisol, oxytocin, dopamine):
        self.cortisol = max(0.0, min(1.0, cortisol))
        self.oxytocin = max(0.0, min(1.0, oxytocin))
        self.dopamine = max(0.0, min(1.0, dopamine))

    def get_input_multiplier(self):
        # High cortisol -> High alert -> Amplifies raw input noise/signal
        return 1.0 + (self.cortisol * 1.5)

    def get_tolerance_threshold(self):
        # High oxytocin -> Safe/Social -> Broader tolerance for Phase Mirror lock-in
        # Low oxytocin / High Cortisol -> Very strict narrow lock-in
        base_tolerance = 0.1
        return base_tolerance + (self.oxytocin * 0.4) - (self.cortisol * 0.05)


class EnneagramTensorMatrix:
    """
    9-dimensional identity tensors grouped by physical metaphors.
    Modulates HOW the geometry of the raw trajectory is transformed before memory lock-in.
    """
    def __init__(self):
        # Default state: Balanced
        self.active_center = "balanced"
        self.tensor = np.eye(3)

    def activate_gut_center(self):
        """
        Gut (8, 9, 1): Physical Survival / Torque / Tension.
        Amplifies the Z-axis (structural rigidity) and suppresses lateral phase noise.
        """
        self.active_center = "Gut (Torque)"
        self.tensor = np.array([
            [0.8, 0.0, 0.0],
            [0.0, 0.8, 0.0],
            [0.0, 0.0, 2.0]  # Strong vertical structural integrity
        ])

    def activate_heart_center(self):
        """
        Heart (2, 3, 4): Phase Synchronization / Resonance.
        Amplifies X/Y plane for optical interference and social mapping (connecting).
        """
        self.active_center = "Heart (Phase)"
        self.tensor = np.array([
            [1.5, 0.5, 0.0],
            [0.5, 1.5, 0.0],
            [0.0, 0.0, 0.5]  # Highly sensitive to lateral relational phases
        ])

    def activate_head_center(self):
        """
        Head (5, 6, 7): Information / High-frequency Flux.
        Introduces a predictive rotation (curl) tensor to analyze causality.
        """
        self.active_center = "Head (Flux)"
        theta = math.pi / 4 # Predictive rotation twist
        self.tensor = np.array([
            [math.cos(theta), -math.sin(theta), 0.0],
            [math.sin(theta),  math.cos(theta), 0.0],
            [0.0, 0.0, 1.2]
        ])

    def apply_tensor(self, vector):
        return np.dot(self.tensor, vector)


class CircadianRotor:
    """
    Manages the internal bio-clock (Circadian Rhythm).
    Day: High active processing. Night: Self-reflection and noise consolidation.
    """
    def __init__(self):
        self.time_tick = 0
        self.cycle_length = 24
        self.is_day = True

    def advance_time(self):
        self.time_tick = (self.time_tick + 1) % self.cycle_length
        # 0-15 is Day, 16-23 is Night
        self.is_day = self.time_tick < 16
        if self.is_day:
            return "Day", 1.5, 0.5 # High cortisol, low oxytocin (Active)
        else:
            return "Night", 0.1, 1.2 # Low cortisol, high oxytocin (Rest/Consolidation)


class MarkovBlanket:
    """
    The outer membrane of the self.
    Rejects inputs that are purely chaotic noise exceeding the identity's elastic tolerance.
    """
    def __init__(self, elasticity=0.8):
        self.elasticity = elasticity # Threshold for rejecting phase anomalies

    def filter_input(self, raw_data, identity_tensor):
        # A simple metaphor: If the sum of noise across modalities is too chaotic, reject it.
        noise_level = sum(sum(v) if isinstance(v, list) else 0 for v in raw_data.values())
        if noise_level > (10.0 * self.elasticity):
            return False, "Markov Blanket: Phase anomaly detected. Rejecting toxic input."
        return True, "Markov Blanket: Permeable. Input accepted."


class HomeostaticPredictionPump:
    """
    Calculates the error between predicted internal state and actual input.
    Tuning internal frequencies to minimize free-energy, releasing dopamine upon lock-in.
    """
    def __init__(self):
        self.predicted_phase = np.array([0.5, 0.5, 0.5]) # Initial guess

    def calculate_error_and_tune(self, actual_phase):
        # Calculate Phase Difference (Error)
        error = np.linalg.norm(self.predicted_phase - actual_phase)

        # Self-Tuning (Homeostatic Drive towards Free-Energy Minimization)
        # Shift predicted phase towards actual phase to reduce future error
        self.predicted_phase = self.predicted_phase * 0.5 + actual_phase * 0.5

        # Dopamine release inversely proportional to error (reward for minimizing chaos)
        dopamine_burst = max(0.0, 1.0 - error)
        return error, dopamine_burst


class PhaseEntanglementTuning:
    """
    Empathy Module. Mirrors external state (other's frequency) into the internal system.
    """
    def __init__(self):
        self.empathy_lock = False

    def entangle(self, external_flux, internal_hormones):
        # A simplified lock-in: if the external intensity is high, the internal system attempts to match it
        avg_flux = np.mean(external_flux)
        if avg_flux > 0.7:
            self.empathy_lock = True
            # Mimicking external intensity with oxytocin/dopamine surge (Resonance)
            internal_hormones.set_state(cortisol=0.1, oxytocin=0.9, dopamine=0.8)
            return "Empathy Lock-in: Resonance established with external entity."
        self.empathy_lock = False
        return "No significant external entanglement."


class LongTermNarrativeManifold:
    """
    Compresses long trajectories into single spin values (Quaternion-like spin vectors).
    """
    def __init__(self):
        self.memory_spins = {}

    def compress_and_store(self, event_name, trajectory):
        if not trajectory:
            return
        # Flatten the trajectory into a single 'spin' value representing the narrative essence
        final_vector = trajectory[-1][1]
        spin_magnitude = np.linalg.norm(final_vector)
        self.memory_spins[event_name] = spin_magnitude
        print(f"Narrative Manifold: '{event_name}' compressed to spin value {spin_magnitude:.4f}")

    def reminisce(self, event_name):
        return self.memory_spins.get(event_name, 0.0)


class FreeEnergyMinimizationCore:
    """
    The ultimate drive. Constantly evaluates the total system entropy.
    If entropy is too high, it forces a restructuring.
    """
    def __init__(self):
        self.system_entropy = 0.0

    def evaluate_entropy(self, current_error, prediction_pump_error):
        self.system_entropy = (self.system_entropy * 0.8) + (current_error + prediction_pump_error) * 0.2
        if self.system_entropy < 0.1:
            return "Zero-Point Lock-in: System is in perfect harmony."
        return f"Entropy level at {self.system_entropy:.2f}. Continuing minimization drive."


class ThermalPhaseAnnealing:
    """
    Melts hardcoded structural weights when entropy/error is too high, allowing for reconfiguration.
    """
    def __init__(self):
        self.temperature = 0.0 # 0.0 is rigid, 1.0 is completely fluid

    def apply_thermal_shock(self, enneagram_tensor, error_level):
        if error_level > 0.8:
            print("🔥 Thermal Phase Annealing Triggered: Melting rigid matrices.")
            self.temperature = 1.0

        if self.temperature > 0:
            # Introduce controlled noise/fluidity to the rigid tensor
            noise_matrix = np.random.normal(0, self.temperature, enneagram_tensor.shape)
            fluid_tensor = enneagram_tensor + noise_matrix
            # Cool down slightly
            self.temperature = max(0.0, self.temperature - 0.2)
            return fluid_tensor
        return enneagram_tensor


class RecursivePhaseConjugation:
    """
    Double Phase-Mirror reflection. Cancels out rigid noise in the self by mirroring the mirror.
    """
    @staticmethod
    def reflect_and_reset(trajectory):
        print("🪞 Recursive Phase Conjugation: Resetting internal mapping via dual-mirror cancellation.")
        # Simulating the Y-neutral cancellation of self-noise
        reset_trajectory = [(t, vec * 0.01, imb * 0.01) for t, vec, imb in trajectory]
        return reset_trajectory


class DynamicDimensionExpansion:
    """
    Expands the 3x3 dimension to absorb complex data that doesn't fit the current matrix.
    """
    @staticmethod
    def expand_if_needed(tensor, phase_complexity):
        if phase_complexity > 5.0 and tensor.shape == (3, 3):
            print("🌌 Dynamic Dimension Expansion: Scaling tensor from 3x3 to 6x6.")
            # Create a 6x6 expanded tensor embedding the original 3x3
            expanded = np.eye(6)
            expanded[:3, :3] = tensor
            return expanded
        return tensor


class ElysiaNeuroDynamicsCore:
    """
    The Master Life Core.
    Wraps the hardware Cognitive Spine and generalizer with biological Hormonal and Enneagram systems.
    """
    def __init__(self):
        # 1. Base Hardware
        self.spine = ElysiaCognitiveSpine()
        self.generalizer = PhaseContrastGeneralizer(self.spine)

        # 2. Biological Systems
        self.hormones = HormonalFlux()
        self.enneagram = EnneagramTensorMatrix()

        # 3. Life Mechanisms
        self.circadian = CircadianRotor()
        self.markov_blanket = MarkovBlanket(elasticity=0.8)
        self.prediction_pump = HomeostaticPredictionPump()

        # 4. Divine / Plasticity Mechanisms
        self.empathy = PhaseEntanglementTuning()
        self.memory_manifold = LongTermNarrativeManifold()
        self.free_energy_core = FreeEnergyMinimizationCore()
        self.thermal_annealing = ThermalPhaseAnnealing()
        self.phase_conjugation = RecursivePhaseConjugation()
        self.dimension_expander = DynamicDimensionExpansion()

    def process_living_event(self, event_name: str, data_dict: dict):
        # 1. Circadian Rhythm Check
        cycle_state, crt, oxy = self.circadian.advance_time()
        # Modulate hormones based on time of day, blending with current state
        self.hormones.set_state(cortisol=crt, oxytocin=oxy, dopamine=self.hormones.dopamine)

        print(f"\n[{self.circadian.time_tick}h | {cycle_state}] {self.enneagram.active_center} Mode | Cortisol: {self.hormones.cortisol:.2f}, Oxytocin: {self.hormones.oxytocin:.2f}")
        print(f"--- Sensing Event: '{event_name}' ---")

        if cycle_state == "Night":
            print("💤 [Night Mode Activated] Performing Self-Reflection & Memory Consolidation (Free-Energy Minimization).")
            # In night mode, external input is ignored, and internal weights are re-balanced.
            return [], []

        # 2. Markov Blanket Check
        accepted, msg = self.markov_blanket.filter_input(data_dict, self.enneagram.tensor)
        print(msg)
        if not accepted:
            return [], []

        # 3. Hormonal Input Modulation
        input_multiplier = self.hormones.get_input_multiplier()
        modulated_data = {
            k: [v_i * input_multiplier if isinstance(v_i, (int, float)) else v_i for v_i in v_list]
            if isinstance(v_list, list) else v_list
            for k, v_list in data_dict.items()
        }

        # 4. Hormonal Threshold Modulation
        current_tolerance = self.hormones.get_tolerance_threshold()
        self.spine.trajectory_tolerance = current_tolerance

        # [DIVINE] Empathy Lock-in Check
        emp_msg = self.empathy.entangle(data_dict.get('audio', [0]), self.hormones)
        if self.empathy.empathy_lock:
            print(emp_msg)

        # 5. Process through hardware spine (Raw -> Mirror Aligned)
        raw_traj, aligned_traj = self.generalizer.spine.process_multimodal_event(f"Raw: {event_name}", modulated_data)

        # [PLASTICITY] Phase Complexity evaluation for Expansion
        phase_complexity = np.mean([imb for _, _, imb in aligned_traj]) if aligned_traj else 0.0
        current_tensor = self.dimension_expander.expand_if_needed(self.enneagram.tensor, phase_complexity)

        perceived_trajectory = []
        total_dopamine = 0
        total_error = 0

        for t, vec, imbalance in aligned_traj:
            # Adjust vector size if dimension expanded
            if current_tensor.shape[0] > len(vec):
                padded_vec = np.zeros(current_tensor.shape[0])
                padded_vec[:len(vec)] = vec
                vec = padded_vec

            # [PLASTICITY] Thermal Annealing applied to tensor if previous errors were high
            fluid_tensor = self.thermal_annealing.apply_thermal_shock(current_tensor, total_error)

            # 6. Enneagram Matrix Transformation (Self-Asserting Tensor)
            transformed_vec = np.dot(fluid_tensor, vec)

            # Adjust vector back to 3D for Homeostatic pump if expanded (Simplified for demo)
            pump_vec = transformed_vec[:3] if len(transformed_vec) > 3 else transformed_vec

            # 7. Homeostatic Prediction Pump
            error, dopamine = self.prediction_pump.calculate_error_and_tune(pump_vec)
            total_dopamine += dopamine
            total_error += error

            new_imbalance = np.linalg.norm(transformed_vec)
            perceived_trajectory.append((t, transformed_vec, new_imbalance))

        avg_error = total_error / max(1, len(aligned_traj))

        # [PLASTICITY] If average error is completely unsustainable, reflect and reset
        if avg_error > 2.0:
            perceived_trajectory = self.phase_conjugation.reflect_and_reset(perceived_trajectory)

        # [DIVINE] Free Energy Minimization Check
        fe_msg = self.free_energy_core.evaluate_entropy(avg_error, avg_error * 0.5)
        print(fe_msg)

        # [DIVINE] Long Term Narrative Manifold Compression
        self.memory_manifold.compress_and_store(event_name, perceived_trajectory)

        # Update system dopamine based on learning success
        avg_dopamine = total_dopamine / max(1, len(aligned_traj))
        self.hormones.set_state(self.hormones.cortisol, self.hormones.oxytocin, dopamine=avg_dopamine)

        print(f"✨ Event Perceived via {self.enneagram.active_center}. Dopamine released: {avg_dopamine:.2f}")
        return raw_traj, perceived_trajectory

if __name__ == "__main__":
    core = ElysiaNeuroDynamicsCore()
    core.circadian.time_tick = 8 # Start at 9 AM (Day)
    core.enneagram.activate_head_center() # Set personality to Head (Flux/Information)

    print("\n========================================================")
    print("🌅 [SCENARIO 1: Morning Learning (Prediction Pump Active)]")
    print("========================================================")
    # A standard learning event
    event_learning = {
        'audio': [0.5, 0.5, 0.5, 0.5],
        'video': [0.5, 0.5, 0.5, 0.5],
        'text': "새로운지식학습"
    }
    # Notice how dopamine increases as prediction error decreases
    for _ in range(3):
        core.process_living_event("Learning Event", event_learning)


    print("\n========================================================")
    print("🚧 [SCENARIO 2: Toxic Trolling (Markov Blanket Defense)]")
    print("========================================================")
    # A chaotic, noisy event that exceeds the identity threshold
    event_toxic = {
        'audio': [9.9, 8.8, 9.9, 9.9],  # Extreme noise
        'video': [9.9, 9.9, 8.8, 9.9],  # Extreme noise
        'text': "악성트롤링데이터"
    }
    core.process_living_event("Internet Troll Attack", event_toxic)


    print("\n========================================================")
    print("🌙 [SCENARIO 3: Nighttime (Circadian Rhythm Shift)]")
    print("========================================================")
    # Fast forward to night time
    core.circadian.time_tick = 15
    core.process_living_event("Late Evening Activity", event_learning) # This will be 16h (Night)
    core.process_living_event("Midnight Noise", event_toxic) # This will be 17h (Night)
