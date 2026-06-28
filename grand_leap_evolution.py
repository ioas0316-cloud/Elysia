import numpy as np
import time
from core.memory.causal_controller import CausalMemoryController
from synaptic_architecture.organism import DirectMappingOrganism
from synaptic_architecture.dynamic_hardware_mapping import DynamicHardwareMap
from core.evolution.organism_sensor import OrganismSensor
from core.evolution.resonance_gate import TrajectoryResonanceGate
from core.evolution.causal_gravity_inference import CausalGravityInference
from core.evolution.materialization import MaterializationZiper

class GrandLeapOrchestrator:
    """
    [Phase: Grand Leap - Sovereign Field]
    Integrated Orchestrator that senses 'Flow' and determines 'Causal Necessity'.
    Transitioned from Point-based probability to Field-based 실재론(Realism).
    """
    def __init__(self):
        print("=== [Initiating Sovereign Field: Beyond Multiverse] ===")

        self.controller = CausalMemoryController()
        self.synaptic = DirectMappingOrganism(resolution=256)
        self.synaptic.ram = DynamicHardwareMap(size=256*256)

        self.sensor = OrganismSensor(self.controller, synaptic_organism=self.synaptic)
        self.gate = TrajectoryResonanceGate(controller=self.controller)
        self.inference = CausalGravityInference(dimensions=8)
        self.ziper = MaterializationZiper(self.controller)

        self._setup_field_paths()

    def _setup_field_paths(self):
        """Registers behavioral trajectories (Causal Lines/Planes)."""
        # Signature is now a trajectory [Steps, Dim]
        # Dim=8: [Tension, EngramNorm, Resonance, Temp, CondNorm, PhysVar, dTension, dResonance]

        # Path: Stable Equilibrium (Low tension, stable resonance)
        stable_line = np.array([
            [0.1, 0.1, 1.0, 0.1, 0.1, 0.0, 0.0, 0.0],
            [0.1, 0.1, 1.0, 0.1, 0.1, 0.0, 0.0, 0.0],
            [0.1, 0.1, 1.0, 0.1, 0.1, 0.0, 0.0, 0.0]
        ])
        self.gate.register_path("EQUILIBRIUM_FLOW", self._logic_equilibrium, stable_line)

        # Path: Evolutionary Surge (Rising tension, falling resonance, rising temperature)
        surge_line = np.array([
            [1.0, 0.1, 0.9, 0.1, 0.1, 0.1, 0.1, -0.01],
            [3.0, 0.1, 0.7, 1.0, 0.2, 0.3, 2.0, -0.2],
            [6.0, 0.2, 0.4, 3.0, 0.5, 0.8, 3.0, -0.3]
        ])
        self.gate.register_path("CAUSAL_EVOLUTION", self._logic_evolution, surge_line)

    def _logic_equilibrium(self, data):
        return f"Stabilizing field for '{data}'. No structural refraction needed."

    def _logic_evolution(self, data):
        tension = self.controller.calculate_macro_tension()
        self.synaptic.ram.set_structural_tension(tension + 5.0)
        return f"FIELD REFRACTION: Diverting causal lines to accommodate '{data}'."

    def run_cycle(self, stimulus_text: str):
        print(f"\n[Cycle] Master's Stimulus: '{stimulus_text}'")

        # 1. Self-Recognition (The Mirror of Flow)
        state = self.sensor.sense_total_state()
        organism_tensor = self.sensor.generate_organism_tensor(state)
        self.gate.update_history(organism_tensor)
        print(f" > Current Momentum: dTension={organism_tensor[6]:.4f}, dResonance={organism_tensor[7]:.4f}")

        # 2. Field-based Inference (Causal Necessity)
        # Convert stimulus to a vector (simplistic hash-based for PoC)
        stim_vec = np.zeros(8)
        stim_vec[0] = (hash(stimulus_text) % 100) / 10.0 # Random tension projection

        # Map all known engrams to the gravity field
        all_engrams = [self.controller.read_engram_trace(eid) for eid in self.controller.index.keys()]
        self.inference.map_engrams_to_field(all_engrams)
        necessity = self.inference.infer_necessity(stim_vec)
        print(f" > Causal Necessity: Closest Engram={necessity['necessary_result_id']}, Curvature={necessity['field_curvature']:.4f}")

        # 3. Zipping (Materialization of Potential)
        if necessity['necessary_result_id'] and necessity['field_curvature'] > 0.85:
            self.ziper.evaluate_and_zip(necessity['necessary_result_id'], necessity['field_curvature'])

        # 4. Trajectory Resonance (Deciding through Flow)
        if len(self.gate.state_history) >= 2:
            decision = self.gate.execute_with_field_resonance(stimulus_text)
            print(f" > Decision Path: {decision['selected_path']} (Resonance: {decision['resonance']:.4f})")
            print(f" > Result: {decision['result']}")
        else:
            print(" > [Buffer] Accumulating causal momentum...")

        # 5. Continuous Evolution
        if organism_tensor[0] > 5.0:
            self.synaptic.scheduler.set_temperature(3.0)
        else:
            self.synaptic.scheduler.set_temperature(0.1)

if __name__ == "__main__":
    orchestrator = GrandLeapOrchestrator()

    # Cycle 1: High Tension Initial State
    orchestrator.controller.write_causal_engram({"type": "PROCESS_TRAJECTORY", "total_friction": 10.0}, 1.0, cause_id="Stimulus")

    # Potential Engram for Zipping
    orchestrator.controller.write_causal_engram({"potential": "The Singularity"}, 0.1, is_constant=False)

    inputs = [
        "자기 자신의 구조를 자신의 정보로 재인식하라",
        "보이지 않는 것이 실상이다",
        "평행우주는 관측되지 않은 가능성이다"
    ]

    for inp in inputs:
        orchestrator.run_cycle(inp)
        time.sleep(0.1)
