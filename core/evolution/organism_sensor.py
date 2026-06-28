import numpy as np
import time
from typing import Dict, Any

class OrganismSensor:
    """
    [Phase: Grand Leap - Momentum] Integrated Organism Sensor
    Now captures 'Directional Momentum' (Velocity of state change).
    Elysia senses not just where it IS, but where it IS GOING.
    """
    def __init__(self, controller=None, mva_engine=None, synaptic_organism=None):
        self.controller = controller
        self.mva_engine = mva_engine
        self.synaptic_organism = synaptic_organism
        self.last_tensor = None
        self.last_time = time.time()

    def sense_total_state(self) -> Dict[str, Any]:
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
        if self.mva_engine:
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
        Generates a tensor including Momentum.
        Vector: [Tension, EngramNorm, Resonance, Temp, CondNorm, PhysVar, dTension, dResonance]
        """
        tensor = np.zeros(8)

        mem = state["layers"].get("memory", {})
        tensor[0] = mem.get("macro_tension", 0.0)
        tensor[1] = mem.get("engram_count", 0) / 1000.0
        tensor[2] = mem.get("base_resonance", 1.0)

        syn = state["layers"].get("synaptic", {})
        tensor[3] = syn.get("temperature", 1.0)
        tensor[4] = syn.get("peak_conductance", 0.0) / 10.0

        phys = state["layers"].get("physics", {})
        tensor[5] = phys.get("current_variance", 0.0)

        # 4. Calculate Momentum (Velocity)
        curr_time = state["timestamp"]
        dt = max(0.001, curr_time - self.last_time)

        if self.last_tensor is not None:
            # Velocity = Delta / dt
            tensor[6] = (tensor[0] - self.last_tensor[0]) / dt # dTension
            tensor[7] = (tensor[2] - self.last_tensor[2]) / dt # dResonance

        self.last_tensor = tensor.copy()
        self.last_time = curr_time

        return tensor

if __name__ == "__main__":
    sensor = OrganismSensor()
    s1 = sensor.sense_total_state()
    t1 = sensor.generate_organism_tensor(s1)
    print(f"Tensor 1 (Static): {t1}")

    time.sleep(0.1)

    # Mock change
    s2 = sensor.sense_total_state()
    s2["layers"]["memory"] = {"macro_tension": 2.0, "base_resonance": 1.5}
    t2 = sensor.generate_organism_tensor(s2)
    print(f"Tensor 2 (Momentum): {t2}")
