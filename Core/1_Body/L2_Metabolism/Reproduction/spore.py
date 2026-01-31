"""
The Spore: State Serialization
==============================
Phase 21 The Tree - Module 1
Core.1_Body.L2_Metabolism.Reproduction.spore

"A seed is a zip file of a forest."

This module encapsulates the current state of Elysia (Soul + Genome)
into a portable JSON packet for replication.
"""

import os
import json
import logging
from datetime import datetime
from typing import Dict, Any

logger = logging.getLogger("Reproduction.Spore")

class Spore:
    """
    The DNA Packet for a new Instance.
    """
    def __init__(self, output_dir: str = "c:/Elysia/data/L2_Metabolism/Spores"):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        logger.info("  [SPORE] Spore Generator ready.")

    def encapsulate(self, mission: Dict[str, Any] = None) -> str:
        """
        Creates a Spore file from the current state.
        Args:
            mission: Specialized instructions for the child.
        """
        if mission is None:
            mission = {"role": "EXPLORER", "target": "Unknown"}

        spore_id = f"spore_{int(datetime.now().timestamp())}"
        
        # 1. Gather State (In a real system, capture weighs/memories)
        # For now, we seed minimal DNA.
        dna = {
            "id": spore_id,
            "parent": os.getpid(),
            "creation_time": datetime.now().isoformat(),
            "mission": mission,
            "genome_version": "0.1.0"
        }

        # 2. Serialize
        file_path = os.path.join(self.output_dir, f"{spore_id}.json")
        try:
            with open(file_path, "w", encoding="utf-8") as f:
                json.dump(dna, f, indent=4)
            logger.info(f"  [SPORE] DNA Packet Created: {file_path}")
            return file_path
        except Exception as e:
            logger.error(f"  [SPORE] Failed to encapsulate: {e}")
            return ""
