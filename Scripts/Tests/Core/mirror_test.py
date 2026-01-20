"""
TEST: MIRROR REFLECTION
=======================
Verifies the Feedback Loop (Phase 18).
"""
import sys
import os
import time

# Add root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from Core.L2_Metabolism.Evolution.action_logger import ActionLogger
from Core.L2_Metabolism.Evolution.evaluator import OutcomeEvaluator

def run_test():
    print("==================================")
    print("   PHASE 18: MIRROR SYSTEM TEST   ")
    print("==================================")

    # 1. Initialize Components
    logger = ActionLogger(log_dir="c:/Elysia/data/Logs/Test")
    judge = OutcomeEvaluator()
    
    # 2. Simulate ACTION 1: Successful File Creation
    print("\n👉 [ACTION 1] Attempting to create a file...")
    intent = "Create 'test.txt'"
    action_id = logger.log_action(intent, "FILE_IO", {"path": "test.txt"})
    
    # Simulate Execution...
    outcome_status = "SUCCESS"
    outcome_data = "File written successfully."
    
    # Evaluate
    score = judge.evaluate(intent, outcome_status, outcome_data)
    logger.log_outcome(action_id, outcome_status, outcome_data, score)
    print(f"   -> Verdict: {score} ({outcome_status})")

    # 3. Simulate ACTION 2: Failed Operation (Exception)
    print("\n👉 [ACTION 2] Attempting a forbidden operation...")
    intent = "Delete System32"
    action_id = logger.log_action(intent, "SYSTEM_OP", {"target": "System32"})
    
    # Simulate Execution...
    try:
        raise PermissionError("Access Denied")
    except Exception as e:
        outcome_status = "ERROR"
        outcome_data = e
        
    # Evaluate
    score = judge.evaluate(intent, outcome_status, outcome_data)
    logger.log_outcome(action_id, outcome_status, outcome_data, score)
    print(f"   -> Verdict: {score} ({outcome_status})")

    print("\n✅ Verification Complete. Check logs in c:/Elysia/data/Logs/Test")

if __name__ == "__main__":
    run_test()
