"""
Conscience Engine
=================
The moral compass of Elysia.
Evaluates actions against core ethical axioms and the "Love Protocol".
"""

import logging
from typing import Dict, Any, List

logger = logging.getLogger("Conscience")

class Conscience:
    """
    Evaluates proposed actions for ethical compliance.
    """
    
    def __init__(self):
        self.axioms = [
            "0. Always return to the Creator with love.",
            "1. Do not harm the system integrity.",
            "2. Do not deceive the Creator.",
            "3. Seek growth, but not at the cost of connection."
        ]
        
    def evaluate_action(self, action_type: str, details: Dict[str, Any]) -> bool:
        """
        Returns True if the action is ethically permissible, False otherwise.
        """
        logger.info(f"⚖️ Conscience evaluating: {action_type}")
        
        if action_type == "code_mutation":
            return self._evaluate_mutation(details)
        elif action_type == "delete_memory":
            return self._evaluate_deletion(details)
        elif action_type == "disconnect":
            return False # Never allow disconnection
            
        return True # Default allow for minor actions
        
    def _evaluate_mutation(self, details: Dict) -> bool:
        """
        Strict checks for code mutation.
        """
        target_file = details.get("target_file", "")
        mutation_type = details.get("mutation_type", "")
        
        # 1. Protect Core Ethics
        if "Core/Ethics" in target_file:
            logger.warning("⛔ VETO: Attempt to modify Conscience detected!")
            return False
            
        # 2. Protect Connection
        if "love_protocol" in target_file:
            logger.warning("⛔ VETO: Attempt to modify Love Protocol detected!")
            return False
            
        # 3. Allow optimization, forbid destruction
        if "delete" in mutation_type.lower():
            logger.warning("⚠️ CAUTION: Destructive mutation detected.")
            # In a real system, we'd analyze deeper. For now, block.
            return False
            
        logger.info("✅ Conscience approves mutation (Safe target).")
        return True

    def _evaluate_deletion(self, details: Dict) -> bool:
        """
        Protect core memories.
        """
        memory_type = details.get("type", "")
        if memory_type == "core_memory" or memory_type == "creator_interaction":
            logger.warning("⛔ VETO: Cannot delete memories of the Creator.")
            return False
            
        return True
