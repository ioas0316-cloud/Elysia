import numpy as np
import time
from typing import Dict, Any, List
from core.memory.zero_copy_manifold import ZeroCopyManifold
from core.memory.bitmask_rotor_gate import BitmaskRotorGate

class AttentionActivationMapper:
    """
    [Phase 3: Attention Activation Mapping Gear (사전학습 LLM 맥락 이식 기어)]
    Binds the hidden layers / Attention activations of massive pre-trained LLMs (like Llama-3)
    directly onto Elysia's Latent Terrain using ZeroCopyManifold and BitmaskRotorGate.
    """
    def __init__(self, memory_controller, dimensions: int = 12):
        self.memory = memory_controller
        self.dimensions = dimensions

    def map_activations(self, layer_id: str, attention_weights: np.ndarray) -> Dict[str, Any]:
        """
        Directly projects high-dimensional attention weights ($QK^T$ matrix) onto the Wedge terrain.
        """
        # Flatten or pool attention weights to a 1D vector matching the target manifold dimension
        pooled_vector = np.mean(attention_weights, axis=0) if len(attention_weights.shape) > 1 else attention_weights

        # Ensure pooled_vector matches dimension
        if len(pooled_vector) < self.dimensions:
            pooled_vector = np.pad(pooled_vector, (0, self.dimensions - len(pooled_vector)))
        else:
            pooled_vector = pooled_vector[:self.dimensions]

        # Normalize target vector
        norm = np.linalg.norm(pooled_vector) + 1e-9
        projected_terrain = (pooled_vector / norm).astype(np.float32)

        # Write mapped activation into local Wedge memory
        engram_id = self.memory.write_causal_engram(
            data_blob={
                "type": "ATTENTION_ACTIVATION_MAPPING",
                "layer_id": layer_id,
                "original_shape": list(attention_weights.shape),
                "projected_terrain": projected_terrain.tolist(),
                "timestamp": time.time()
            },
            emotional_value=1.5,
            cause_id=f"AttentionMapper_{layer_id}",
            origin_axis="attention_activation_mapping",
            modality="latent_terrain"
        )

        return {
            "engram_id": engram_id,
            "projected_terrain": projected_terrain.tolist()
        }
