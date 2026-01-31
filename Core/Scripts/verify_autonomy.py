"""
Verification Script: The First Mutation (Autonomy)
==================================================
Tests if Elysia can securely self-modify.
1. Creates a 'Sick File' (Inefficient code).
2. Triggers the Growth Node logic (Mocked).
3. Verifies Consent and Mutation.
"""

import sys
import os
import logging
import time
from unittest.mock import MagicMock

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.L6_Structure.Autonomy.autonomy_protocol import AutonomyProtocol
from Core.L6_Structure.Autonomy.self_maintenance_hub import SelfMaintenanceHub, SystemDiagnosis

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VerifyAutonomy")

def test_autonomy_protocol():
    logger.info("\n[TEST] Autonomy Protocol (Consent)")
    protocol = AutonomyProtocol()
    
    # Case 1: High Coherence (Stable) -> Consent
    field_stable = {"autonomy_level": 2, "coherence": 0.9}
    consent_stable = protocol.check_consent("Plan A", field_stable)
    logger.info(f"Stable State: {consent_stable}")
    
    if consent_stable['consent']:
        logger.info("PASS: Consent Granted in Stable State.")
    else:
        logger.error("FAIL: Consent Denied in Stable State.")
        return False
        
    # Case 2: Low Coherence (Panic) -> Deny
    field_panic = {"autonomy_level": 2, "coherence": 0.1}
    consent_panic = protocol.check_consent("Plan B", field_panic)
    logger.info(f"Panic State: {consent_panic}")
    
    if not consent_panic['consent'] and "Instability" in consent_panic['reason']:
        logger.info("PASS: Consent Denied in Panic State.")
    else:
        logger.error("FAIL: Protocol failed to block Panic Mutation.")
        return False
        
    return True

def test_mutation_simulation():
    logger.info("\n[TEST] Mutation Simulation")
    
    # 1. Create Dummy Issue
    dummy_file = "c:/Elysia/Scripts/dummy_mutation_target.py"
    with open(dummy_file, "w") as f:
        f.write("def slow_code():\n    for i in range(1000000): pass\n")
        
    # 2. Mock Hub
    hub = SelfMaintenanceHub()
    # Mock diagnose to return health < 1.0 and point to dummy_file
    diagnosis = SystemDiagnosis(health_score=0.8, bottlenecks=[f"{dummy_file} (CPU)"])
    hub.diagnose = MagicMock(return_value=diagnosis)
    
    # Mock propose_fix to return a plan
    mock_plan = MagicMock()
    mock_plan.target_file = dummy_file
    hub.propose_fix = MagicMock(return_value=mock_plan)
    
    # Mock execute
    hub.execute_with_consent = MagicMock(return_value=True)
    
    # 3. Simulate Growth Vibration Logic
    protocol = AutonomyProtocol()
    field = {"autonomy_level": 2, "coherence": 0.8, "thought_log": []}
    
    # --- Logic Replay ---
    diag = hub.diagnose()
    if diag.health_score < 1.0:
        logger.info("Issue Detected.")
        plan = hub.propose_fix(dummy_file)
        if plan:
            consent = protocol.check_consent(plan, field)
            if consent['consent']:
                logger.info("Consent Received. Executing...")
                success = hub.execute_with_consent(plan, consent=True)
                if success:
                    logger.info("Mutation Executed.")
                    
    # Verification
    if hub.execute_with_consent.called:
        logger.info("PASS: Mutation Logic executed successfully.")
        return True
    else:
        logger.error("FAIL: Mutation not executed.")
        return False

if __name__ == "__main__":
    if test_autonomy_protocol() and test_mutation_simulation():
        print("\nALL AUTONOMY TESTS PASSED.")
    else:
        print("\nVERIFICATION FAILED.")
