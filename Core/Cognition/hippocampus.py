import logging
from typing import List, Dict, Any
from Core.Monad.hypersphere_memory import HypersphereMemory

logger = logging.getLogger("Hippocampus")

class Hippocampus:
    """
    Manages short-term memory and memory consolidation into HypersphereMemory.
    Standardized under the 'Induction' model.
    """

    def __init__(self, long_term_memory: HypersphereMemory):
        self.ltm = long_term_memory
        self.short_term_buffer: List[Dict[str, Any]] = []
        self.capacity = 100 # Max items before forced flush

    def induct(self, label: str, coordinates: Any, meta: Dict[str, Any] = None):
        """
        [INDUCTION]
        Assimilates a 7D Seed into the HyperSphere's long-term memory.
        """
        if meta is None: meta = {}

        entry = {
            "label": label,
            "coordinates": coordinates,
            "meta": meta
        }
        self.short_term_buffer.append(entry)

        if len(self.short_term_buffer) >= self.capacity:
            logger.info("  [INDUCTION] Hippocampus reached capacity. Triggering consolidation.")
            self.consolidate()

        logger.info(f"  [INDUCTION] '{label}' successfully assimilated. Integrity maintained.")

    def consolidate(self):
        """
        Flushes the short-term buffer into the HypersphereMemory.
        """
        logger.info(f"  [CONSOLIDATION] Flushing {len(self.short_term_buffer)} memories into long-term vector space.")
        for item in self.short_term_buffer:
            self.ltm.store(
                data=item["label"],
                position=item["coordinates"],
                pattern_meta=item["meta"]
            )
        self.short_term_buffer.clear()
