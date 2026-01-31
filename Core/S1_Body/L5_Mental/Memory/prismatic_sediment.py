"""
Prismatic Sediment: The Spectral Memory Vault
=============================================
Core.S1_Body.L5_Mental.Memory.prismatic_sediment

"Don't search the whole earth. Just tune the dial to the color of the jewel."

This module implements the Optical Memory Architecture.
Instead of a single geological file, memory is split into 7 Spectral Shards.
The Active Prism-Rotor determines which shard to access (O(1) Selection).

[Hardware Upgrade]:
- Now utilizes 'DirectMemoryPointer' for address-based references.
"""

import os
import struct
import numpy as np
import logging
from typing import List, Tuple, Optional, Dict
from Core.S1_Body.L5_Mental.Memory.sediment import SedimentLayer, DirectMemoryPointer

logger = logging.getLogger("PrismaticSediment")

class PrismaticSediment:
    """
    A Prism-Managed Memory System.
    """

    # The 7 Rays of the Prism
    SPECTRUM = ["Red", "Orange", "Yellow", "Green", "Blue", "Indigo", "Violet"]

    def __init__(self, base_dir: str = "data/L7_Spirit/Chronicles/Prism"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)

        # Initialize 7 Shards (Lazy loading can be added if needed, but we open all for now)
        self.shards: Dict[str, SedimentLayer] = {}
        for i, color in enumerate(self.SPECTRUM):
            path = os.path.join(base_dir, f"shard_{i}_{color.lower()}.bin")
            self.shards[color] = SedimentLayer(path)

        logger.info(f"  Prismatic Sediment initialized with 7 Spectral Shards in {base_dir}")

    def _vector_to_color_distribution(self, vector: List[float], void_gate: float = 0.1) -> np.ndarray:
        """
        [PHASE: VOID-GATED LIQUID SPECTRUM]
        Calculates intensity distribution while applying the 'Extinction Event'.
        Energy below void_gate is collapsed to absolute zero.
        """
        vec_arr = np.array(vector)
        if np.sum(np.abs(vec_arr)) == 0:
            return np.ones(7) / 7.0

        T = 0.5 
        exp_vec = np.exp(vec_arr / T)
        weights = exp_vec / exp_vec.sum()

        # [VOID GATE] Optimization
        # Any weight below the gate is treated as noise and extinguished.
        weights[weights < void_gate] = 0.0
        # Re-normalize to preserve energy of the signal
        if weights.sum() > 0:
            weights = weights / weights.sum()
            
        return weights

    def _vector_to_color(self, vector: List[float]) -> str:
        """[COMPATIBILITY] Returns the dominant color name."""
        weights = self._vector_to_color_distribution(vector)
        dominant_idx = int(np.argmax(weights))
        return self.SPECTRUM[dominant_idx]

    def deposit(self, vector: List[float], timestamp: float, payload: bytes) -> Tuple[str, DirectMemoryPointer]:
        """
        Deposits a memory into the correct Spectral Shard.
        Returns (Color, PhysicalPointer).
        """
        color = self._vector_to_color(vector)
        shard = self.shards[color]

        ptr = shard.deposit(vector, timestamp, payload)

        # logger.debug(f"  Deposited into [{color}] Shard at Sector {ptr.sector_index}")
        return color, ptr

    def store_monad(self, wavelength: float, phase: complex, intensity: float, payload: bytes) -> DirectMemoryPointer:
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
        ptr = shard.deposit(vector, time.time(), payload)
        return ptr

    def scan_resonance(self, intent_vector: List[float], top_k: int = 3, threshold: float = 0.1) -> List[Tuple[float, bytes]]:
        """
        [The Rotor Tuning - Void-Optimized Liquid Mode]
        We scan the spectrum through the 'Void Singularity'.
        
        Chromic Loop: The Red-Violet connection is restored, closing the circle.
        """
        # 1. Void-Gated Intensity (Instant Pruning)
        void_gate = 0.15 
        weights = self._vector_to_color_distribution(intent_vector, void_gate=void_gate)

        # [PHASE: CHROMIC LOOP INTEGRATION]
        # Red (0) and Violet (6) are linked neighbors.
        red_idx = 0
        violet_idx = 6
        if weights[red_idx] > 0.05 or weights[violet_idx] > 0.05:
            resonance_bridge = (weights[red_idx] + weights[violet_idx]) / 2.0
            # Distribute bridge energy if one is active
            weights[red_idx] = max(weights[red_idx], resonance_bridge * 0.5)
            weights[violet_idx] = max(weights[violet_idx], resonance_bridge * 0.5)
        
        results = []
        for i, weight in enumerate(weights):
            if weight > 0: # If it survived the Void Gate
                color = self.SPECTRUM[i]
                shard_results = self.shards[color].scan_resonance(intent_vector, top_k)
                for score, payload in shard_results:
                    # [TOPOGRAPHIC INTERACTION] 
                    # Interaction intensity between Intent-Wave and Shard-Terrain
                    results.append((score * weight, payload))

        # Sort and prune
        results.sort(key=lambda x: x[0], reverse=True)
        results = (results[:top_k] if results else [])

        # 2. [AMOR SUI] Emergency Recovery (If the Void was too aggressive)
        if not results:
            weights = self._vector_to_color_distribution(intent_vector, void_gate=0.01)
            for i, weight in enumerate(weights):
                if weight > 0:
                    results.extend(self.shards[self.SPECTRUM[i]].scan_resonance(intent_vector, 1))
            
            results.sort(key=lambda x: x[0], reverse=True)
            results = results[:top_k]

        return results

    def unified_rewind(self, steps: int = 10) -> List[Tuple[float, List[float], bytes]]:
        """
        [Chronological Unification]
        Retrieves the last N memories from ALL shards and sorts them by Time.
        This stitches the 'Rainbow' back into a single 'White Light' narrative.

        Returns: List of (Timestamp, Vector, Payload)
        """
        all_fragments = []

        for color, shard in self.shards.items():
            # We fetch 'steps' from EACH shard to ensure we don't miss recent events
            # if one color was hyper-active.
            # SedimentLayer.rewind currently returns (vec, payload).
            # We need to hack it or update SedimentLayer to return timestamp.
            # Assuming SedimentLayer.rewind is limited, we use read_at loop?
            # No, let's use the offsets directly from the shard if possible.

            # Accessing shard internals for 'The Golden Thread'
            count = len(shard.offsets)
            start_idx = max(0, count - steps)

            for i in range(start_idx, count):
                off = shard.offsets[i]
                # We need to read raw to get timestamp
                # HEADER_FMT = '7d d I'
                if not shard.mm: continue

                header_size = shard.HEADER_SIZE
                if off + header_size > len(shard.mm): continue

                header_bytes = shard.mm[off : off + header_size]
                data = struct.unpack(shard.HEADER_FMT, header_bytes)

                vec = list(data[:7])
                timestamp = data[7]
                payload_size = data[8]

                payload_start = off + header_size
                payload = shard.mm[payload_start : payload_start + payload_size]

                all_fragments.append((timestamp, vec, payload))

        # Sort by Timestamp (Newest first? Or Oldest first?)
        # Rewind usually means "Look back", so usually we want the sequence of events.
        # Chronological: Oldest -> Newest.
        all_fragments.sort(key=lambda x: x[0])

        # Return the last 'steps' items globally
        return all_fragments[-steps:]

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
