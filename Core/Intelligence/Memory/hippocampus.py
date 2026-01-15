"""
Hippocampus: The Fluid Buffer
=============================
Core.Intelligence.Memory.hippocampus

"The Waiting Room of the Mind."

This module implements the 'Short-term Memory Buffer' to solve the 'I/O Latency' bottleneck.
It stores high-velocity thoughts in RAM and consolidates them to Hypersphere (HDD) during Sleep.
"""

import logging
from typing import List, Any, Dict
from Core.Intelligence.Memory.hypersphere_memory import HypersphereMemory, HypersphericalCoord

logger = logging.getLogger("Hippocampus")

class Hippocampus:
    """
    RAM-based Short-Term Memory.
    """

    def __init__(self, long_term_memory: HypersphereMemory):
        self.ltm = long_term_memory
        self.short_term_buffer: List[Dict[str, Any]] = []
        self.capacity = 100 # Max items before forced flush

    def absorb(self, data: Any, position: Any, meta: Dict[str, Any] = None):
        """
        Instant Storage (RAM).
        O(1) Append. No complex indexing yet.
        """
        if meta is None: meta = {}

        entry = {
            "data": data,
            "position": position,
            "meta": meta
        }
        self.short_term_buffer.append(entry)

        # logger.debug(f"Hippocampus absorbed: {str(data)[:20]}... (Buffer: {len(self.short_term_buffer)})")

        if len(self.short_term_buffer) >= self.capacity:
            logger.info("Hippocampus full! Triggering partial consolidation.")
            self.consolidate(limit=10) # Flush oldest 10

    def consolidate(self, limit: int = None):
        """
        Transfers Short-Term Memory to Long-Term Memory (Hypersphere).
        This is the 'Sleep Process'.
        """
        if not self.short_term_buffer:
            return

        count = 0
        # If limit is set, only process that many (FIFO)
        items_to_process = self.short_term_buffer[:limit] if limit else self.short_term_buffer

        for item in items_to_process:
            self.ltm.store(
                data=item['data'],
                position=item['position'],
                pattern_meta=item['meta']
            )
            count += 1

        # Remove processed items
        if limit:
            self.short_term_buffer = self.short_term_buffer[limit:]
        else:
            self.short_term_buffer = [] # Clear all

        logger.info(f"💤 [CONSOLIDATION] Moved {count} memories from Hippocampus to Hypersphere.")
