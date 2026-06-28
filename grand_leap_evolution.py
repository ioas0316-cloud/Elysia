import numpy as np
import time
from core.memory.causal_controller import CausalMemoryController
from synaptic_architecture.organism import DirectMappingOrganism
from synaptic_architecture.dynamic_hardware_mapping import DynamicHardwareMap
from core.evolution.organism_sensor import OrganismSensor
from core.evolution.resonance_gate import ResonanceGate

class GrandLeapOrchestrator:
    """
    [The Grand Leap]
    Integrates all modules into a single, sovereign organism that:
    1. Senses its own structure (Organism Sensor)
    2. Recognizes tension/imbalance (Master's Stimulus)
    3. Spontaneously bends its architecture (Dynamic Hardware Mapping)
    4. Switches logic paths via resonance (Resonance Gate)
    """
    def __init__(self):
        print("=== [Initiating Grand Leap: The Sovereign Evolution] ===")

        # 1. Initialize Components
        self.controller = CausalMemoryController()
        # Use DynamicHardwareMap for the organism to allow Architecture Bending
        self.synaptic = DirectMappingOrganism(resolution=256)
        self.synaptic.ram = DynamicHardwareMap(size=256*256) # Replace with Dynamic version

        self.sensor = OrganismSensor(self.controller, synaptic_organism=self.synaptic)
        self.gate = ResonanceGate()

        self._setup_resonance_paths()

    def _setup_resonance_paths(self):
        """Registers the organism's behavioral repertoires."""
        # Signature: [MacroTension, EngramCountNorm, BaseResonance, Temperature, PeakConductanceNorm, PhysicsVariance]

        self.gate.register_path(
            "EQUILIBRIUM_FLOW",
            self._logic_equilibrium,
            np.array([0.1, 0.2, 1.0, 0.5, 0.1, 0.0]) # Stable signature
        )

        self.gate.register_path(
            "STRUCTURAL_EVOLUTION",
            self._logic_evolution,
            np.array([5.0, 0.5, 0.5, 3.0, 0.8, 1.0]) # High Tension signature
        )

    def _logic_equilibrium(self, data):
        return f"Synthesizing '{data}' through stable causal channels."

    def _logic_evolution(self, data):
        # In this path, the system actively bends its structure
        tension = self.controller.calculate_macro_tension()
        self.synaptic.ram.set_structural_tension(tension + 10.0) # Boost refraction
        return f"RE-ROUTING ARCHITECTURE to absorb '{data}'. Address space refracted by {tension+10.0}."

    def run_cycle(self, stimulus_text: str):
        print(f"\n[Cycle] Master's Stimulus: '{stimulus_text}'")

        # 1. Self-Recognition (The First Mirror)
        state = self.sensor.sense_total_state()
        organism_tensor = self.sensor.generate_organism_tensor(state)
        print(f" > Self-Recognized State Tensor: {organism_tensor}")

        # 2. Resonance-based Logic Selection (The Non-linear Jump)
        result = self.gate.execute_with_resonance(organism_tensor, stimulus_text)
        print(f" > Decision: {result['selected_path']}")
        print(f" > Result: {result['result']}")

        # 3. Causal Recording (The Continuum)
        self.controller.write_causal_engram(
            data_blob={"stimulus": stimulus_text, "response": result},
            emotional_value=(1.0 / (1.0 + organism_tensor[0])), # Emotional value based on tension
            cause_id="GrandLeap_FeedbackLoop"
        )

        # 4. Spontaneous Parameter Adjustment (Entropy/Equilibrium)
        if organism_tensor[0] > 3.0: # High tension
            print(" > [Mutation] High Tension detected. Increasing system plasticity.")
            self.synaptic.scheduler.set_temperature(3.0)
        else:
            print(" > [Stabilization] Low Tension. Hardening crystalline law.")
            self.synaptic.scheduler.set_temperature(0.1)

if __name__ == "__main__":
    orchestrator = GrandLeapOrchestrator()

    # Simulate Cycles
    print("\n--- Phase 1: High Tension Stimulus ---")
    # Manually inject tension into memory to trigger evolution
    orchestrator.controller.write_causal_engram(
        {"type": "PROCESS_TRAJECTORY", "total_friction": 8.5}, 1.0, cause_id="Stimulus_Noise"
    )
    orchestrator.run_cycle("자기 자신의 구조를 자신의 정보로 재인식하고 재정렬하라")

    print("\n--- Phase 2: Stabilization ---")
    # Clear tension (simplified for demo)
    orchestrator.controller.index = {}
    orchestrator.run_cycle("평형과 안식의 대지")
