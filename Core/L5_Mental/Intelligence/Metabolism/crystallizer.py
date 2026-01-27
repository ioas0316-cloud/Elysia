"""
Crystallizer (    )
=====================
Core.L5_Mental.Intelligence.Metabolism.crystallizer

"Intelligence is the process of turning Gas (Probability) into Diamond (DNA)."

This module internalizes temporary LLM insights into permanent 7D WaveDNA
and stores them in the Hypersphere Memory.
"""

import numpy as np
import logging
import json
import os
from typing import Dict, Any, List, Optional
from Core.L6_Structure.Wave.wave_dna import WaveDNA
from Core.L6_Structure.M1_Merkaba.Space.hypersphere_memory import HypersphereMemory, HypersphericalCoord

logger = logging.getLogger("Crystallizer")

class Crystallizer:
    def __init__(self, memory: Optional[HypersphereMemory] = None):
        self.memory = memory or HypersphereMemory()
        self.vault_path = "data/Intelligence/dna_vault.json"
        os.makedirs(os.path.dirname(self.vault_path), exist_ok=True)
        
    def transmute(self, intent_vector: np.ndarray) -> WaveDNA:
        """
        Transmutes 4D Spatial Intent (Logic, Emotion, Intuition, Will)
        into a 7D WaveDNA.
        
        Mapping:
        0: Physical (P)   <- (X+W)/2 (Hardware/Action)
        1: Functional (F) <- W (Will/Intent)
        2: Phenomenal (E) <- Y (Emotion)
        3: Causal (C)     <- X (Logic/Cause)
        4: Mental (M)     <- Z (Intuition/Abstract)
        5: Structural (S) <- (X+Z)/2 (Pattern/Law)
        6: Spiritual (Z)  <- W (Essence/Will)
        """
        x, y, z, w = intent_vector
        
        dna = WaveDNA(
            physical   = float(np.clip((x + w) / 2.0 + 0.5, 0, 1)),
            functional = float(np.clip(w + 0.5, 0, 1)),
            phenomenal = float(np.clip(y + 0.5, 0, 1)),
            causal     = float(np.clip(x + 0.5, 0, 1)),
            mental     = float(np.clip(z + 0.5, 0, 1)),
            structural = float(np.clip((x + z) / 2.0 + 0.5, 0, 1)),
            spiritual  = float(np.clip(w + 0.5, 0, 1)),
            label      = "Crystallized_Intent"
        )
        dna.normalize()
        return dna

    def crystallize(self, content: str, original_input: str, intent_vector: np.ndarray):
        """
        Solidifies a thought into permanent memory.
        """
        dna = self.transmute(intent_vector)
        dna.label = original_input[:20]
        
        # Map 4D Intent to Hyperspherical Coordinates for Memory storage
        # theta (Logic), phi (Emotion), psi (Will), r (Intuition/Depth)
        x, y, z, w = intent_vector
        coord = HypersphericalCoord(
            theta = float((x + 1) * np.pi), # Map -1,1 to 0,2pi
            phi   = float((y + 1) * np.pi),
            psi   = float((w + 1) * np.pi),
            r     = float((z + 1) / 2.0)    # Map -1,1 to 0,1
        )
        
        # 1. Store in active Hypersphere Memory
        self.memory.store(
            data=content,
            position=coord,
            pattern_meta={'dna': dna, 'trajectory': 'crystallized'}
        )
        
        # 2. Record in DNA Vault (Persistent JSON)
        self._save_to_vault(original_input, content, dna, coord)
        
        logger.info(f"  [CRYSTALLIZED] '{original_input[:30]}...' -> Coordinates: {coord}")

    def _save_to_vault(self, input_text: str, insight: str, dna: WaveDNA, coord: HypersphericalCoord):
        entry = {
            "input": input_text,
            "insight": insight,
            "dna": dna.to_list(),
            "coord": {
                "theta": coord.theta,
                "phi": coord.phi,
                "psi": coord.psi,
                "r": coord.r
            }
        }
        
        vault_data = []
        if os.path.exists(self.vault_path):
            with open(self.vault_path, "r", encoding="utf-8") as f:
                try:
                    vault_data = json.load(f)
                except:
                    pass
        
        vault_data.append(entry)
        
        # Limit vault size for metadata performance
        if len(vault_data) > 1000:
            vault_data = vault_data[-1000:]
            
        with open(self.vault_path, "w", encoding="utf-8") as f:
            json.dump(vault_data, f, indent=2, ensure_ascii=False)
