"""
The Diary: Action Logger
========================
Phase 18 The Mirror - Module 1
Core.S1_Body.L2_Metabolism.Evolution.action_logger

"To remember is to learn. To forget is to repeat."

This module records every specific action taken by the system,
preserving the 'Intent' and the 'Context' for later reflection.
"""

import json
import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger("Evolution.Diary")

class ActionLogger:
    """
    The Chronicler of Deeds.
    Writes structured JSONL logs to data/L1_Foundation/M4_Logs/action_history.jsonl
    """
    def __init__(self, log_dir: str = "c:/Elysia/data/L6_Structure/Logs"):
        self.log_file = os.path.join(log_dir, "action_history.jsonl")
        os.makedirs(log_dir, exist_ok=True)
        logger.info(f"  [DIARY] Opening logbook at: {self.log_file}")

    def log_action(self, 
                   intent: str, 
                   action_type: str, 
                   details: Dict[str, Any], 
                   expected_outcome: Optional[str] = None) -> str:
        """
        Records an action *before* or *during* execution.
        Returns a unique Action ID (timestamp based) for linking results later.
        """
        timestamp = datetime.now().isoformat()
        action_id = f"ACT_{int(datetime.now().timestamp() * 1000)}"
        
        entry = {
            "id": action_id,
            "timestamp": timestamp,
            "phase": "EXECUTION",
            "intent": intent,
            "action_type": action_type,
            "details": details,
            "expected_outcome": expected_outcome or "Success"
        }
        
        self._append_to_log(entry)
        logger.info(f"  [ACTION] {action_type}: {intent} (ID: {action_id})")
        return action_id

    def log_outcome(self, 
                    action_id: str, 
                    result_status: str, 
                    result_data: Any, 
                    score: float) -> None:
        """
        Records the result *after* execution.
        """
        timestamp = datetime.now().isoformat()
        
        entry = {
            "id": action_id,
            "timestamp": timestamp,
            "phase": "REFLECTION",
            "result_status": result_status, # SUCCESS / FAILURE / ERROR
            "result_data": str(result_data),
            "score": score
        }
        
        self._append_to_log(entry)
        logger.info(f"  [REFLECTION] {action_id} -> {result_status} (Score: {score})")

    def _append_to_log(self, entry: Dict[str, Any]):
        try:
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"  [DIARY] Failed to write log: {e}")
