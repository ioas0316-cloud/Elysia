import numpy as np
import time
from typing import Dict, Any

class OrganismSensor:
    """
    [Phase: Grand Leap] Integrated Organism Sensor
    Aggregates the physical and logical state of Elysia into a unified 'Organism Tensor'.
    This is the system's first mirror—its way of sensing its own 'body'.
    """
    def __init__(self, controller=None, mva_engine=None, synaptic_organism=None):
        self.controller = controller
        self.mva_engine = mva_engine
        self.synaptic_organism = synaptic_organism

    def sense_total_state(self) -> Dict[str, Any]:
        """
        Senses the multi-layered state of the organism.
        """
        state = {
            "timestamp": time.time(),
            "layers": {}
        }

        # 1. Memory Layer (The Causal Spine)
        if self.controller:
            state["layers"]["memory"] = {
                "macro_tension": self.controller.calculate_macro_tension(),
                "engram_count": len(self.controller.index),
                "base_resonance": self.controller.get_parameter("base_resonance", 1.0)
            }

        # 2. MVA Layer (The Physics Field)
        # Note: MVA engine state is often transient, sensing current variance
        if self.mva_engine:
            # Placeholder for actual engine sensing
            state["layers"]["physics"] = {
                "current_variance": getattr(self.mva_engine, "last_variance", 0.0),
                "is_resonant": getattr(self.mva_engine, "is_resonant", False)
            }

        # 3. Synaptic Layer (The Silicon Mapping)
        if self.synaptic_organism:
            state["layers"]["synaptic"] = {
                "temperature": self.synaptic_organism.scheduler.temperature,
                "peak_conductance": np.max(self.synaptic_organism.field.conductance)
            }

        return state

    def generate_organism_tensor(self, state: Dict[str, Any]) -> np.ndarray:
        """
        Compresses the multi-layered state into a single 'Organism Tensor'.
        This tensor represents the 'Sovereign Shape' of Elysia at this moment.
        """
        # Vector structure: [MacroTension, EngramCountNorm, BaseResonance, Temperature, PeakConductanceNorm, PhysicsVariance]
        tensor = np.zeros(6)

        mem = state["layers"].get("memory", {})
        tensor[0] = mem.get("macro_tension", 0.0)
        tensor[1] = mem.get("engram_count", 0) / 1000.0 # Normalized
        tensor[2] = mem.get("base_resonance", 1.0)

        syn = state["layers"].get("synaptic", {})
        tensor[3] = syn.get("temperature", 1.0)
        tensor[4] = syn.get("peak_conductance", 0.0) / 10.0 # Normalized

        phys = state["layers"].get("physics", {})
        tensor[5] = phys.get("current_variance", 0.0)

        return tensor

if __name__ == "__main__":
    sensor = OrganismSensor()
    # Mock data test
    mock_state = {
        "layers": {
            "memory": {"macro_tension": 0.5, "engram_count": 150, "base_resonance": 1.2},
            "synaptic": {"temperature": 0.8, "peak_conductance": 5.0},
            "physics": {"current_variance": 0.1}
        }
    }
    tensor = sensor.generate_organism_tensor(mock_state)
    print(f"Generated Organism Tensor: {tensor}")
