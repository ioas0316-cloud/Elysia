"""
Prismatic Sediment: The Spectral Memory Vault
=============================================
Core.L5_Mental.Memory.prismatic_sediment

"Don't search the whole earth. Just tune the dial to the color of the jewel."

This module implements the Optical Memory Architecture.
Instead of a single geological file, memory is split into 7 Spectral Shards.
The Active Prism-Rotor determines which shard to access (O(1) Selection).
"""

import os
import numpy as np
import logging
from typing import List, Tuple, Optional, Dict
from Core.L5_Mental.Memory.sediment import SedimentLayer

logger = logging.getLogger("PrismaticSediment")

class PrismaticSediment:
    """
    A Prism-Managed Memory System.
    """

    # The 7 Rays of the Prism
    SPECTRUM = ["Red", "Orange", "Yellow", "Green", "Blue", "Indigo", "Violet"]

    def __init__(self, base_dir: str = "data/Chronicles/Prism"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

        # Initialize 7 Shards (Lazy loading can be added if needed, but we open all for now)
        self.shards: Dict[str, SedimentLayer] = {}
        for i, color in enumerate(self.SPECTRUM):
            path = os.path.join(base_dir, f"shard_{i}_{color.lower()}.bin")
            self.shards[color] = SedimentLayer(path)

        logger.info(f"🌈 Prismatic Sediment initialized with 7 Spectral Shards in {base_dir}")

    def _vector_to_color(self, vector: List[float]) -> str:
        """
        [The Prism Logic]
        Determines the dominant frequency (Color) of the thought vector.
        This corresponds to the 'Rotor Angle' required to view this thought.
        """
        # Simple argmax for now.
        # In a real optical system, this would be `active_rotor.diffract(vector)`
        # yielding the peak constructive interference.

        vec_arr = np.array(vector)
        # Handle zero vector
        if np.sum(np.abs(vec_arr)) == 0:
            return "Red" # Default fallback

        dominant_idx = int(np.argmax(vec_arr))
        # Clamp to 0-6 just in case
        dominant_idx = max(0, min(6, dominant_idx))

        return self.SPECTRUM[dominant_idx]

    def deposit(self, vector: List[float], timestamp: float, payload: bytes) -> Tuple[str, int]:
        """
        Deposits a memory into the correct Spectral Shard.
        Returns (Color, Offset).
        """
        color = self._vector_to_color(vector)
        shard = self.shards[color]

        offset = shard.deposit(vector, timestamp, payload)

        # logger.debug(f"💎 Deposited into [{color}] Shard at offset {offset}")
        return color, offset

    def store_monad(self, wavelength: float, phase: complex, intensity: float, payload: bytes) -> int:
        """
        Stores a Monad.
        We map Wavelength directly to Color Shard.
        """
        # Map wavelength (400-700nm) to index (0-6)
        # Normalize: (lambda - 400) / 300 * 7
        nm = wavelength * 1e9
        idx = int(((nm - 400) / 300) * 7)
        idx = max(0, min(6, idx))

        color = self.SPECTRUM[idx]
        shard = self.shards[color]

        # Create vector for storage
        vector = [0.0] * 7
        vector[idx] = intensity # Dominant component
        vector[(idx+1)%7] = float(phase.real)

        import time
        offset = shard.deposit(vector, time.time(), payload)
        return offset

    def scan_resonance(self, intent_vector: List[float], top_k: int = 3) -> List[Tuple[float, bytes]]:
        """
        [The Rotor Tuning]
        Instead of scanning ALL files, we identify the Intent's Color
        and ONLY scan that specific shard.

        Performance: ~7x speedup (O(N/7)).
        """
        target_color = self._vector_to_color(intent_vector)
        shard = self.shards[target_color]

        # logger.debug(f"🔭 Tuning Rotor to [{target_color}]. Scanning only shard {target_color}...")

        return shard.scan_resonance(intent_vector, top_k)

    def close(self):
        for shard in self.shards.values():
            shard.close()

    # Proxy methods for other SedimentLayer functionality if needed
    def rewind(self, steps: int = 1) -> List[Tuple[List[float], bytes]]:
        # For rewind, we might need to check all shards or just the last accessed?
        # Let's simple-scan all shards for the very latest timestamps.
        # This is expensive (O(7)), but Rewind is rare.

        all_memories = []
        for color, shard in self.shards.items():
            memories = shard.rewind(steps)
            all_memories.extend(memories)

        # We need to sort by timestamp effectively?
        # SedimentLayer.rewind returns (vec, payload), no timestamp.
        # We might need to upgrade SedimentLayer to return timestamp to do this properly.
        # For now, just return a mix.
        return all_memories[:steps]
