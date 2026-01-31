"""
The Akashic Records (Spatial Memory)
====================================
Core.S1_Body.L4_Causality.World.Architecture.spatial_memory

"The blueprint is not lost; it is etched into the ether."

This module implements the storage interface for the Quantum Pop-Up system.
It utilizes the Sediment Layer (Hypersphere) to persist architectural crystallizations.
"""

import time
import json
import logging
import os
from typing import Tuple, Dict, Any, List
from Core.S1_Body.L5_Mental.Memory.sediment import SedimentLayer

logger = logging.getLogger("SpatialMemory")

class SpatialMemory:
    """
    Interface for the Hypersphere's Spatial Memory Sector.
    """
    def __init__(self):
        # Ensure directory exists
        os.makedirs("data/L5_Mental/M1_Memory", exist_ok=True)
        self.layer = SedimentLayer("data/L5_Mental/M1_Memory/spatial_sediment.bin")

    def crystallize(self, structure_obj: str, metadata: Dict[str, Any], vector: List[float] = None) -> int:
        """
        Persists a 3D structure into the Hypersphere.
        Returns the Memory Address (Pointer).
        """
        if vector is None:
            vector = [0.5] * 7 # Default neutral vector

        # Pack payload: JSON Metadata + Separator + OBJ Data
        payload_dict = {
            "meta": metadata,
            "obj": structure_obj
        }
        # Use simple JSON serialization
        payload_bytes = json.dumps(payload_dict).encode('utf-8')

        timestamp = time.time()
        address = self.layer.deposit(vector, timestamp, payload_bytes)

        logger.info(f"  Crystallized Structure at Address {address} (Size: {len(payload_bytes)} bytes)")
        return address

    def recall(self, address: int) -> Tuple[str, Dict[str, Any]]:
        """
        Retrieves a structure from the Hypersphere.
        Returns (OBJ_String, Metadata).
        """
        result = self.layer.read_at(address)
        if not result:
            logger.warning(f"Memory Access Fault at {address}")
            return "", {}

        vector, payload_bytes = result
        try:
            payload_dict = json.loads(payload_bytes.decode('utf-8'))
            return payload_dict["obj"], payload_dict.get("meta", {})
        except Exception as e:
            logger.error(f"Failed to parse memory at {address}: {e}")
            return "", {}

    def close(self):
        self.layer.close()
