"""
Merkaba Simulator (The Ghost in the Machine)
============================================
Core.Merkaba.simulator

"Simulation is the bridge between Pattern and Principle."

This tool performs 'Virtual Firing' of neural clusters to understand
the causal logic gates extracted during the Logic Autopsy.
"""

import numpy as np
import json
import os
import struct
import logging
from typing import Dict, Any, List
from Core.Merkaba.portal import MerkabaPortal
from Core.Merkaba.safetensors_decoder import SafetensorsDecoder

logger = logging.getLogger("Elysia.Merkaba.Simulator")

class RotorSimulator:
    def __init__(self, model_index_path: str):
        self.index_path = model_index_path
        with open(model_index_path, 'r') as f:
            self.index = json.load(f)
        self.weight_map = self.index.get("weight_map", {})
        self.base_dir = os.path.dirname(model_index_path)

    def ignite_hub(self, tensor_name: str, input_vector: np.ndarray) -> np.ndarray:
        """
        Simulates the causal response of a specific hub to an input vector.
        (O(1) memory usage via mmap)
        """
        shard_name = self.weight_map.get(tensor_name)
        if not shard_name:
            raise ValueError(f"Hub '{tensor_name}' not found in HyperSphere.")
            
        shard_path = os.path.join(self.base_dir, shard_name)
        
        with MerkabaPortal(shard_path) as portal:
            header = SafetensorsDecoder.get_header(portal)
            header_size = struct.unpack("<Q", portal.mm[:8])[0]
            meta = header.get(tensor_name)
            
            rel_offsets = meta["data_offsets"]
            start_off, end_off = SafetensorsDecoder.get_absolute_offset(header_size, rel_offsets)
            length = end_off - start_off
            shape = meta["shape"]
            
            # Perception: View the weight without loading
            weights = portal.read_view(start_off, length, dtype=np.float16)
            weights = weights.reshape(shape)
            
            # Dimension Adaptation (Universal Probe fitting)
            # weights shape is usually [out_features, in_features]
            in_features = shape[1] if weights.ndim > 1 else shape[0]
            
            adapted_input = input_vector.astype(np.float32)
            if len(adapted_input) > in_features:
                adapted_input = adapted_input[:in_features]
            elif len(adapted_input) < in_features:
                adapted_input = np.pad(adapted_input, (0, in_features - len(adapted_input)))

            # Causal Propagation: Dot product (The Logic Gate)
            if weights.ndim > 1:
                output = np.dot(adapted_input, weights.T.astype(np.float32))
            else:
                output = adapted_input * weights.astype(np.float32)
            
            del weights
            return output

if __name__ == "__main__":
    import os, json, struct
    print("Rotor Simulator: Causal engine ready.")
