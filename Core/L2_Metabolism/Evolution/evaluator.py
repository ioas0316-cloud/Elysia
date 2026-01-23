"""
The Judge: Outcome Evaluator
============================
Phase 18 The Mirror - Module 2
Core.L2_Metabolism.Evolution.evaluator

"Not all actions are born equal. Some bear fruit, others bear thorns."

This module assesses the immediate result of an action.
It acts as the critic, assigning a scalar 'fitness score' to behavior.
"""

import logging
from typing import Any, Dict

logger = logging.getLogger("Evolution.Judge")

class OutcomeEvaluator:
    """
    The Objective Critic.
    """
    def __init__(self):
        logger.info(f"   [JUDGE] Court is in session.")

    def evaluate(self, intent: str, result_status: str, result_data: Any) -> float:
        """
        Calculates a fitness score (-1.0 to +1.0).
        
        Args:
            intent: What triggered the action.
            result_status: SUCCESS / FAILURE / ERROR.
            result_data: The return value or exception object.
        """
        
        # 1. Base Score on Status
        if result_status == "ERROR":
            score = -1.0
            judgment = "CRITICAL FAILURE"
        elif result_status == "FAILURE":
            score = -0.5
            judgment = "FAILURE"
        elif result_status == "SUCCESS":
            score = 0.5 # Default success is good, but not perfect
            judgment = "SUCCESS"
        else:
            score = 0.0
            judgment = "UNKNOWN"

        # 2. Refinement based on Exceptions (Implicit Pain)
        if isinstance(result_data, Exception):
            score = -1.0
            judgment = f"EXCEPTION: {type(result_data).__name__}"

        # 3. Refinement based on User Feedback (Explicit Pleasure/Pain)
        # TODO: Implement Natural Language Sentiment Analysis on result_data if string

        logger.info(f"   [JUDGE] Verdict on '{intent}': {judgment} (Score: {score})")
        return score