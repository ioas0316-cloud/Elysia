import numpy as np
import time
from typing import Dict, List, Any, Optional

class DynamicAxisSprouter:
    """
    [Phase 3: Dynamic Axis Sprouting Gear (자율적 관점 축 분화 기어)]
    When Elysia observes two ideas/concepts that share high resonance but have a
    residual variance (Tension Variance) that cannot be fully explained by existing axes,
    it dynamically sprouts (spawns) a new dimension/axis (e.g. e_love, e_passion, e_lightning).

    This replaces fixed dimensions with a self-evolving Hilbert space that expands
    as Elysia experiences and observes more deep metaphoric connections.
    """
    def __init__(self, memory_controller):
        self.memory = memory_controller
        self.sprouted_axes_count = 0

    def evaluate_and_sprout(self, label1: str, label2: str, sameness_info: dict) -> Optional[Dict[str, Any]]:
        """
        Evaluates sameness variance. If there is significant unexplained variance
        despite overall high resonance, it sprouts a new conceptual axis representing the
        unique boundary (metaphor) between the two concepts.
        """
        variance = sameness_info.get("sameness_variance", 0.0)
        min_diff = sameness_info.get("min_difference", 1.0)

        # We sprout a new axis when there is a delicate tension/unexplained divergence
        if variance > 0.05 and min_diff < 0.5:
            self.sprouted_axes_count += 1
            axis_name = f"axis_sprouted_{label1}_{label2}_{self.sprouted_axes_count}"

            # The new axis vector is constructed by orthogonalizing the difference
            # or projecting the best sameness axis into a higher-dimensional representation
            best_axis = np.array(sameness_info.get("best_sameness_axis", []), dtype=np.float32)
            if len(best_axis) == 0:
                best_axis = np.random.randn(12).astype(np.float32)
                best_axis /= np.linalg.norm(best_axis)

            # Create a mutated high-dimensional axis representation
            sprouted_vector = (best_axis * 0.8) + (np.random.randn(len(best_axis)) * 0.2)
            sprouted_vector = sprouted_vector / (np.linalg.norm(sprouted_vector) + 1e-9)

            # Record the sprouted axis as a physical truth in cognitive_params
            params = self.memory.cognitive_params
            if "sprouted_dimensions" not in params:
                params["sprouted_dimensions"] = {}

            params["sprouted_dimensions"][axis_name] = {
                "vector": sprouted_vector.tolist(),
                "genesis_time": time.time(),
                "parent_concepts": [label1, label2],
                "variance_resolved": float(variance)
            }
            self.memory._save_cognitive_params()

            # Write an engram representing this new conceptual axis (Cognitive Spatula)
            engram_id = self.memory.write_causal_engram(
                data_blob={
                    "type": "SPROUTED_COGNITIVE_AXIS",
                    "axis_name": axis_name,
                    "vector": sprouted_vector.tolist(),
                    "parent_concepts": [label1, label2],
                    "unexplained_variance": float(variance),
                    "description": f"Sprouted new conceptual axis representing the unique metaphorical bridge between '{label1}' and '{label2}'."
                },
                emotional_value=variance * 10.0,
                cause_id=f"AxisSprout_{label1}_{label2}",
                origin_axis="conceptual_sprouting",
                modality="axis_genesis"
            )

            print(f"[AxisSprout] SPROUTED new conceptual axis '{axis_name}' with engram [{engram_id}]")
            return {
                "axis_name": axis_name,
                "vector": sprouted_vector.tolist(),
                "engram_id": engram_id
            }

        return None
