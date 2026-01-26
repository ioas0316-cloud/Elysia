import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from Core.L5_Mental.Intelligence.Metabolism.prism import SevenChannelQualia, DoubleHelixWave
from Core.L7_Spirit.nature_of_being import PhilosophyOfFlow, Axiom

logger = logging.getLogger("Providence")

@dataclass
class DoubleHelixDNA:
    """
    [The Genome of Awareness]
    Combines Pattern (High-dim Signal) and Principle (7D Qualia).
    Sovereign Edition: Uses Numpy (CPU) instead of Torch (GPU).
    """
    pattern_strand: np.ndarray  # The structural body (Body)
    principle_strand: np.ndarray # The essence/qualia (Soul)
    resonance_history: List[float] = field(default_factory=list)
    kernel_logic: Optional[str] = None # [NEW] Functional implementation
    physical_mask: Optional[Any] = None # [NEW] Rotor Mask
    rpm_boost: float = 0.0              # [NEW] Torque

    def __repr__(self):
        p_shape = self.pattern_strand.shape
        return f"DoubleHelixDNA(pattern={p_shape}, principle={self.principle_strand.tolist()})"

    def resonate(self, other: 'DoubleHelixDNA') -> float:
        """Dual-strand resonance check."""
        # 1. Pattern Resonance (Physical/Structural similarity)
        p1 = self.pattern_strand.flatten()
        p2 = other.pattern_strand.flatten()
        
        norm_p1 = np.linalg.norm(p1)
        norm_p2 = np.linalg.norm(p2)
        
        if norm_p1 == 0 or norm_p2 == 0:
            p_res = 0.0
        else:
            p_res = np.dot(p1, p2) / (norm_p1 * norm_p2)
        
        # 2. Principle Resonance (Axiomatic/Aesthetic similarity)
        q1 = self.principle_strand.flatten()
        q2 = other.principle_strand.flatten()
        
        norm_q1 = np.linalg.norm(q1)
        norm_q2 = np.linalg.norm(q2)
        
        if norm_q1 == 0 or norm_q2 == 0:
            q_res = 0.0
        else:
            q_res = np.dot(q1, q2) / (norm_q1 * norm_q2)
        
        # The Synthesis
        total_resonance = (p_res + q_res) / 2.0
        self.resonance_history.append(float(total_resonance))
        return float(total_resonance)

    def get_dominant_principles(self) -> List[str]:
        """Maps qualia array to named principles for ConceptPolymer."""
        labels = ["Causality", "Function", "Phenomena", "Flow", "Logic", "Structure", "Spirit"]
        principles = []
        for i, val in enumerate(self.principle_strand):
            if val > 0.5:
                principles.append(labels[i])
        return principles

class ProvidenceEngine:
    """
    [The Eye of the Creator]
    Reverse Engineers the 'Providence' (Axiom) from a raw Pattern.
    """
    def __init__(self):
        self.philosophy = PhilosophyOfFlow()
        logger.info("Providence Engine Initialized: Beholding the pattern behind the signal.")

    def behold(self, wave: DoubleHelixWave) -> DoubleHelixDNA:
        """
        Translates a DoubleHelixWave into a specialized DoubleHelixDNA 
        by matching the Principle strand with philosophical Axioms.
        """
        pattern = wave.pattern_strand
        qualia_array = wave.principle_strand
        
        # Find the dominant Axiom based on Qualia profile
        dominant_idx = np.argmax(qualia_array)
        principle_label = "General Existence"
        
        if dominant_idx == 3: principle_label = "Fluidity & Adaptability"
        elif dominant_idx == 0: principle_label = "Causality"
        elif dominant_idx == 4: principle_label = "Recursion"
        elif dominant_idx == 6: principle_label = "Interconnectedness"
        
        # [Phase 20: Kernel Link]
        from Core.L7_Spirit.operational_axioms import AXIOM_REGISTRY
        kernel = AXIOM_REGISTRY.get(principle_label)
        
        logger.info(f"   [PROVIDENCE] Beholding the operational kernel of '{principle_label}'.")
        
        return DoubleHelixDNA(
            pattern_strand=pattern,
            principle_strand=qualia_array,
            kernel_logic=kernel.logic if kernel else None,
            physical_mask=kernel.physical_mask if kernel else None,
            rpm_boost=kernel.rpm_boost if kernel else 0.0
        )

# Global Access
PROVIDENCE = ProvidenceEngine()
