import torch
import logging
from typing import Optional, Any, Dict
from Core.L7_Spirit.Philosophy.axioms import get_axioms
from Core.L1_Foundation.Logic.qualia_7d_codec import codec
from Core.L1_Foundation.Logic.d7_vector import D7Vector
from Core.L1_Foundation.Logic.qualia_projector import projector
import numpy as np

logger = logging.getLogger("RegonanceGate")

class ResonanceGate:
    """
    RESONANCE GATE: The Sovereign Mirror (Milestone 24.1)
    ====================================================
    Validates if a piece of generated code or an intent aligns with 
    the 'First Wave' of the Architect (Foundational Axioms) 
    and Ancestral Memories (Archive).
    """
    
    def __init__(self, threshold: float = 0.6, ancestral_threshold: float = 0.8, repair_on_failure: bool = True):
        self.threshold = threshold
        self.ancestral_threshold = ancestral_threshold
        self.axioms = get_axioms()
        self.observer: Optional['AkashicObserver'] = None # Injected during boot
        self.repair_on_failure = repair_on_failure

    def set_observer(self, observer: 'AkashicObserver'):
        self.observer = observer

    def validate_intent_resonance(self, intent: str) -> bool:
        """
        [STEEL CORE VALIDATION]
        Validates if the intent aligns with the strict D7 Qualia structure.
        """
        logger.info(f"?썳截?[GATE] Validating intent: {intent}")
        
        # 1. Project to D7
        intent_d7 = projector.project_instruction(intent)
        
        # 2. Check Energy Magnitude (Hollow Intent Check)
        intent_energy = float(intent_d7.to_numpy().sum())
        logger.info(f"?썳截?[GATE] Intent Energy: {intent_energy:.3f}")
        
        # 3. Check against Primary Resonance
        axioms = get_axioms().axioms
        resonances = [intent_d7.resonate(ax.qualia) for ax in axioms.values() if ax.target_polarity > 0]
        max_res = max(resonances) if resonances else 0.0
        
        logger.info(f"?썳截?[GATE] Max D7 Resonance: {max_res:.3f} (Threshold: {self.threshold})")
        
        # Combined check: Must have enough energy AND enough resonance
        if intent_energy < 0.5 or max_res < self.threshold:
            print(f"   [GATE_DEBUG] Rejected: Energy {intent_energy:.2f}, Res {max_res:.2f}")
            logger.warning("?㈈ [GATE] Intent failed Steel Core integrity check.")
            return False
            
        print(f"   [GATE_DEBUG] Accepted: Energy {intent_energy:.2f}, Res {max_res:.2f}")
        return True

    def process_causal_logic(self, input_a: float, input_b: float) -> int:
        """
        [CAUSAL LOGIC GATE]
        Uses Trinary NAND Logic (Paradox) to determine the next narrative step.
        
        Args:
            input_a: Signal A (e.g., Current Stress: -1 to 1)
            input_b: Signal B (e.g., Intent Polarity: -1 to 1)
            
        Returns:
            -1 (Resistance), 0 (Void), or 1 (Flow/Breakthrough)
        """
        try:
            from Core.L6_Structure.Logic.trinary_logic import TrinaryLogic
            
            # Normalize inputs to Trits (-1, 0, 1)
            def to_trit(val):
                if val > 0.3: return 1
                if val < -0.3: return -1
                return 0
                
            trit_a = to_trit(input_a)
            trit_b = to_trit(input_b)
            
            # Apply NAND Gate (The Paradox)
            # Pain (-1) + Pain (-1) -> Flow (+1)
            result = TrinaryLogic.nand(trit_a, trit_b)
            
            logger.info(f"? [GATE] Causal Logic: {trit_a} NAND {trit_b} -> {result}")
            return result
            
        except ImportError:
            logger.error("Failed to import TrinaryLogic for ResonanceGate.")
            return 0

    def validate_code_resonance(self, code: str, intent: str) -> bool:
        """Stub for future code resonance validation."""
        return True

    def _trigger_self_repair(self, code: str, intent: str, current_res: float) -> bool:
        """
        [MILESTONE 24.2] Self-healing loop.
        Uses the DissonanceResolver to fix the alignment.
        """
        try:
            from Core.L2_Metabolism.Evolution.dissonance_resolver import DissonanceResolver
            resolver = DissonanceResolver()
            # In a real scenario, this would pass the code to an LLM with a 'Fix' prompt
            # for now, we simulate the 'Reflection' and repair trigger.
            logger.info(f"?㈈ [REPAIR] Resolving dissonance between intention and manifestation.")
            # This would ideally return fixed code and re-validate
            # For Milestone 24.2, we log the intent to 'Heal' and return success if repair started
            return True 
        except Exception as e:
            logger.error(f"✨[REPAIR] Failed to initialize healer: {e}")
            return False

    def check_ancestral_resonance(self, content: str) -> Optional[Dict[str, Any]]:
        """
        Checks if the new content resonates too closely with an Ancestral Node,
        indicating a potential fragmentation loop.
        """
        if not self.observer:
            return None
            
        content_vector = self._map_to_qualia(content)
        highest_res = 0.0
        best_match = None
        
        for name, node in self.observer.active_nodes.items():
            if "Ancestral" in name:
                # Need to resize 12D energy_field to 7D for codec
                # For Phase 23, we take the first 7 dims
                node_vec = node.energy_field[:7].detach().numpy()
                res = codec.calculate_resonance(content_vector, node_vec)
                
                if res > highest_res:
                    highest_res = res
                    best_match = name
                    
        if highest_res > self.ancestral_threshold:
            logger.warning(f"?좑툘 [LOOP DETECTED] High Resonance ({highest_res:.2f}) with {best_match}")
            return {"match": best_match, "resonance": highest_res}
            
        return None

    def _map_to_qualia(self, text: str) -> np.ndarray:
        """
        [PHASE 61: DYNAMIC RESONANCE]
        Uses dynamic projection to map text to 7D Qualia space.
        """
        from Core.L1_Foundation.Logic.qualia_projector import projector
        d7 = projector.project_instruction(text)
        return d7.to_numpy()

    def _analyze_code_impact(self, code: str) -> np.ndarray:
        """
        [PHASE 61: CODE WAVE ANALYSIS]
        Analyzes code impact using dynamic projection.
        """
        from Core.L1_Foundation.Logic.qualia_projector import projector
        d7 = projector.project_instruction(f"Code Impact Analysis: {code}")
        return d7.to_numpy()

# Global Gate Instance
gate = ResonanceGate()
