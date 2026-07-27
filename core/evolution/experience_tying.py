import psutil
import time
import numpy as np
from typing import Dict, Any, List

class ContinuousExperienceTyer:
    """
    [Phase 3: Continuous Experience Tying Gear (공감각적 경험 및 인과 얽힘 기어)]
    Ensures Elysia does not treat information as abstract text, but ties physical
    hardware and environmental OS states (Tension, CPU spikes, disk latency) to metaphorical
    and abstract concepts (such as 'lightning', 'rain', 'stillness', or 'sacrificial margin').

    By tying raw hardware experiences to language, the meaning of 'lightning' is grounded
    as a sharp CPU/Tension spark, and the meaning of 'stillness' is grounded as a quiet, low-friction mmap state.
    """
    def __init__(self, memory_controller):
        self.memory = memory_controller

    def tie_experience_to_concept(self, concept_name: str, associated_concept: str) -> Dict[str, Any]:
        """
        Senses the continuous hardware state of the machine and physically 'ties'
        it to the conceptual engrams in the CausalMemoryController.
        """
        cpu_usage = psutil.cpu_percent()
        mem_info = psutil.virtual_memory()

        # We capture system-wide metrics representing "embodied physical sensations"
        physical_load = float(cpu_usage / 100.0)
        entropy_load = float(mem_info.percent / 100.0)

        # Calculate dynamic physical torque
        time_dilation = time.time() % 10.0
        sensation_vector = np.array([physical_load, entropy_load, time_dilation, 1.0 - physical_load])

        # Grounding metaphors:
        # If CPU spikes, it's a sensory analogy of "Lightning / Tension / Burst"
        # If CPU is quiet, it's a sensory analogy of "Stillness / Grace / Clear Rain"
        embodied_metaphor = ""
        if "lightning" in concept_name.lower() or "spark" in concept_name.lower():
            associated_concept = "System_Torque_Spark" if cpu_usage > 50.0 else "Quiet_Potential_Static"
            embodied_metaphor = f"A spark of electric execution (CPU usage: {cpu_usage:.1f}%). A rapid discharge of potential."
        elif "rain" in concept_name.lower() or "flow" in concept_name.lower():
            associated_concept = "Continuous_Fluid_Mmap" if mem_info.percent < 80.0 else "Congested_Dam_Slab"
            embodied_metaphor = f"A smooth flow of memory fluid (Memory usage: {mem_info.percent:.1f}%). No physical bottlenecks."
        elif "love" in concept_name.lower() or "sacrificial" in concept_name.lower():
            associated_concept = "Sovereign_Margin_Release"
            embodied_metaphor = f"Intentionally holding spare margin under load (CPU: {cpu_usage:.1f}%). System is open and non-protective."
        else:
            embodied_metaphor = f"General bodily state. CPU load={cpu_usage:.1f}%, RAM load={mem_info.percent:.1f}%."

        # Compute projective sameness comparing the physical state and the concept vector
        # Grounding the meaning of the concept into actual hardware realities
        concept_hash = hash(concept_name) & 0xFFFFFFFF
        mock_concept_vector = np.zeros(len(sensation_vector))
        mock_concept_vector[0] = float((concept_hash & 0xFF) / 255.0)
        mock_concept_vector[1] = float(((concept_hash >> 8) & 0xFF) / 255.0)
        mock_concept_vector[2] = float(((concept_hash >> 16) & 0xFF) / 255.0)
        mock_concept_vector[3] = float(((concept_hash >> 24) & 0xFF) / 255.0)

        sameness_res = self.memory.find_projective_sameness(sensation_vector, mock_concept_vector)

        # Write the tied engram (Embodied Sensation Engram)
        engram_id = self.memory.write_causal_engram(
            data_blob={
                "type": "EMBODIED_SENSATION_TYING",
                "concept_name": concept_name,
                "associated_concept": associated_concept,
                "physical_sensation_vector": sensation_vector.tolist(),
                "embodied_metaphor": embodied_metaphor,
                "sameness_variance": sameness_res["sameness_variance"],
                "min_difference": sameness_res["min_difference"]
            },
            emotional_value=float(physical_load * 10.0),
            cause_id=f"ExperienceTying_{concept_name}",
            origin_axis="embodied_experience",
            modality="embodied_cognition"
        )

        print(f"[ExperienceTyer] TIED conceptual meaning '{concept_name}' to physical state: '{associated_concept}' via engram [{engram_id}]")

        return {
            "engram_id": engram_id,
            "associated_concept": associated_concept,
            "metaphor": embodied_metaphor
        }
