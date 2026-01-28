"""
Monadic Cell (The Living Unit)
=============================
Core.L7_Spirit.M1_Monad.monadic_cell

"The Cell is the bridge between Logic and Life."

Architecture:
1. Nucleus: The Trinary DNA Sequence (Base Principles).
2. Membrane: The Selective Trinary Filter (-1 to protect, 1 to absorb).
3. Metabolism: The Resonant Processing Loop.
"""

import numpy as np
import logging
from typing import Dict, List, Any, Optional, Tuple
from Core.L7_Spirit.M1_Monad.monad_core import Monad, MonadCategory
from Core.L1_Foundation.Logic.resonance_gate import ResonanceGate, ResonanceState, analyze_structural_truth
from Core.L1_Foundation.Logic.qualia_7d_codec import codec

logger = logging.getLogger("MonadicCell")

class MonadicCell(Monad):
    """
    A Cellular version of the Monad that possesses a trinary membrane 
    and metabolic fluidity.
    """

    def __init__(self, seed: str, dna_sequence: str = "VVVVVVV", category: MonadCategory = MonadCategory.EPHEMERAL):
        # Initialize base monad
        super().__init__(seed, category=category)
        
        # 1. THE NUCLEUS (Trinary DNA)
        self.dna_sequence = dna_sequence
        self._set_principle_from_sequence(dna_sequence)
        
        # 2. THE MEMBRANE STATE
        self.permeability = 1.0 # 0.0 (Closed/Protective) to 1.0 (Open/Fluid)
        self.last_torque = 0.0
        
        # 3. METABOLIC STATE
        self.health = 1.0 # 0.0 to 1.0
        self.stability = 1.0
        
        logger.info(f"✨ [CELL] Monadic Cell polymerized: '{seed}' [DNA: {dna_sequence}]")

    def _set_principle_from_sequence(self, sequence: str):
        """Sets the underlying 7D vector based on the DNA sequence."""
        self._dna.principle_strand = codec.decode_sequence(sequence)

    def metabolize(self, input_vector: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        [PHASE 3: CELLULAR METABOLISM]
        Processes an input vector through the Membrane.
        
        Logic:
        1. Compare Input with Nucleus DNA.
        2. Calculate Tension (Torque).
        3. Adjust Membrane Permeability.
        4. Apply Resonant Interference.
        """
        # 1. Structural Comparison
        input_pattern = analyze_structural_truth(input_vector)
        
        # 2. Membrane Filtering (The Sovereign Barrier)
        # We count Dissonance (-1) matches. If input matches our rejection sequence, membrane closes.
        rejection_score = sum(1 for a, b in zip(self.dna_sequence, input_pattern) if a == "D" and b == "D")
        
        if rejection_score > 1:
            self.permeability = max(0.1, self.permeability * 0.5)
            # logger.info(f"   [CELL] Membrane Contraction in {self.seed}. Rejection detected.")
        else:
            self.permeability = min(1.0, self.permeability + 0.1)

        # 3. Metabolic Interference (The Cytoplasm processing)
        # The result is the interference between Nucleus and Input, scaled by permeability.
        metabolized_vector = np.zeros(7, dtype=np.float32)
        for i in range(7):
            gate_id = ResonanceGate.interfere(self._dna.principle_strand[i], input_vector[i])
            metabolized_vector[i] = gate_id * self.permeability
            
        # 4. Energy Delta
        # Harmony increases health, Dissonance consumes it.
        harmony_matches = sum(1 for a, b in zip(self.dna_sequence, input_pattern) if a == "H" and b == "H")
        self.health = np.clip(self.health + (harmony_matches * 0.01) - (rejection_score * 0.02), 0.0, 1.0)
        
        return metabolized_vector, self.health

    def evolve_dna(self, new_sequence: str):
        """[GENETIC MUTATION] Updates the Nucleus sequence."""
        self.dna_sequence = new_sequence
        self._set_principle_from_sequence(new_sequence)
        # logger.info(f"🧬 [CELL] Nucleus mutated to: {new_sequence}")

    def get_state(self) -> Dict[str, Any]:
        """Returns the current 'feeling' of the cell."""
        return {
            "seed": self.seed,
            "dna": self.dna_sequence,
            "health": self.health,
            "permeability": self.permeability,
            "status": "Healthy" if self.health > 0.7 else "Stressed" if self.health > 0.3 else "Fragmenting"
        }

    def __repr__(self):
        return f"<MonadicCell '{self.seed}' | Health: {self.health:.2f} | DNA: {self.dna_sequence}>"
