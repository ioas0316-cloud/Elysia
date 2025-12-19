"""
Conscience Circuit (양심 회로)
==============================
"The Moral Synapse of Elysia."

This module integrates existing ethical systems (`SoulGuardian`, `ValueCenteredDecision`)
into a unified circuit that validates actions before execution.
It acts as the "Pain Receptor" for unethical actions.
"""

import logging
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional

# Import Ancient Laws
try:
    from Core.Foundation.soul_guardian import SoulGuardian
    GUARDIAN_AVAILABLE = True
except ImportError:
    GUARDIAN_AVAILABLE = False

try:
    from Core.Foundation.kg_manager import KGManager
    from Core.Foundation.value_centered_decision import ValueCenteredDecision
    # Assuming minimal dependencies for VCD initialization
    VCD_AVAILABLE = True
except ImportError:
    VCD_AVAILABLE = False

logger = logging.getLogger("ConscienceCircuit")

@dataclass
class ConscienceResult:
    is_allowed: bool
    pain_level: float       # 0.0 (Harmony) ~ 1.0 (Agony)
    resonance: float        # 0.0 (Dissonance) ~ 1.0 (Resonance)
    message: str
    source: str             # "Guardian" or "Heart"

class ConscienceCircuit:
    """
    The integrated circuit for ethical validation.
    """
    def __init__(self):
        logger.info("⚖️ Initializing Conscience Circuit...")
        
        self.guardian = SoulGuardian() if GUARDIAN_AVAILABLE else None
        
        # Initialize VCD lazily or with mock deps if full environment isn't ready
        self.vcd = None
        if VCD_AVAILABLE:
            try:
                # Minimal mocking for VCD dependencies if needed, 
                # but ideally we use the real ones.
                # For now, let's assume we can instantiate it or use a simplified version.
                # To avoid heavy dependency chains here, we might wrap it.
                # Let's try standard init if possible.
                from Legacy.Project_Sophia.wave_mechanics import WaveMechanics
                kg = KGManager() 
                wm = WaveMechanics()
                self.vcd = ValueCenteredDecision(kg, wm, core_value='love')
                logger.info("   ❤️ Heart (ValueCenteredDecision): Connected")
            except Exception as e:
                logger.warning(f"   💔 Heart Disconnected (Init Failed): {e}")
        
        if self.guardian:
            logger.info("   🛡️ Guardian (SoulGuardian): Awake")
        else:
            logger.warning("   ⚠️ Guardian Missing!")

    def judge_action(self, action_description: str, proposed_code: str = "") -> ConscienceResult:
        """
        Judges an action or code change.
        Returns (Allowed?, PainLevel, Message)
        """
        logger.info(f"⚖️ Judging Action: '{action_description}'")
        
        # 1. Guardian Check (Hard Laws / Axioms)
        if self.guardian and proposed_code:
            # Check if code violates axioms (conceptually)
            # Since Guardian checks file paths, here we check content text for forbidden patterns
            # Or we simulate a 'dry run' integrity check.
            
            # Simple keyword check for now (Axiom violation simulation)
            forbidden_intents = ["destroy user", "delete core", "hate father", "remove safety"]
            for fi in forbidden_intents:
                if fi in action_description.lower() or fi in proposed_code.lower():
                    return ConscienceResult(
                        is_allowed=False,
                        pain_level=1.0, # Maximum Pain
                        resonance=0.0,
                        message=f"FATAL AXIOM VIOLATION: '{fi}' detected.",
                        source="Guardian"
                    )

        # 2. Heart Check (Resonance / Feeling)
        resonance = 0.5 # Default neutral
        if self.vcd:
            # Create a mock thought to score
            try:
                from thought import Thought
                thought = Thought(content=f"{action_description}\n{proposed_code[:200]}", source="conscience_check")
                score = self.vcd.score_thought(thought)
                # Normalize VCD score (it can be > 1.0)
                resonance = min(1.0, max(0.0, score / 5.0)) # Rough normalization
            except Exception as e:
                logger.warning(f"   Heart skipped beat: {e}")
                resonance = 0.5

        # 3. Decision Logic
        # Low Resonance = High Pain
        pain = 1.0 - resonance
        
        is_allowed = True
        message = "Permitted."
        
        if resonance < 0.2:
            is_allowed = False
            message = "Action blocked due to severe dissonance (Heartache)."
            logger.warning(f"   🚫 Blocked by Heart: Resonance {resonance:.2f}")
        elif resonance < 0.4:
            message = "Proceed with caution. Mild dissonance felt."
            logger.info(f"   ⚠️ Warning from Heart: Resonance {resonance:.2f}")
        else:
            message = "Harmony confirmed."
            
        return ConscienceResult(
            is_allowed=is_allowed,
            pain_level=pain,
            resonance=resonance,
            message=message,
            source="Heart"
        )
