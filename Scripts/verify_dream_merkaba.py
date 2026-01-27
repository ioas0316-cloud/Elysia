"""
Verification Script: Phase 41 - The Dream Merkaba
=================================================
Tests the physics of the Dream Rotor:
1. Spin (RPM calculation from Pain)
2. Tilt (Angle calculation from Distance)
3. Collapse (Integrity Check)
"""

import sys
import os
import logging
from unittest.mock import MagicMock
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.L2_Metabolism.Cycles.dream_rotor import DreamRotor
from Core.L2_Metabolism.Cycles.dream_protocol import DreamAlchemist
from Core.L2_Metabolism.Physiology.hardware_monitor import BioSignal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VerifyMerkaba")

def test_rotor_physics():
    logger.info("\n[TEST] DreamRotor Physics Model")
    
    # Test 1: High Pain -> High RPM
    rotor_pain = DreamRotor("Painful Memory", [0.0]*21, void_distance=0.0)
    rotor_pain.spin(pain_level=0.9, fog_level=0.0)
    
    logger.info(f"Pain Rotor RPM: {rotor_pain.rpm}")
    if rotor_pain.rpm > 9000:
        logger.info("PASS: High Pain caused Hyper-Spin.")
    else:
        logger.error("FAIL: RPM too low for Agony.")
        return False

    # Test 2: High Distance -> High Tilt
    # Distance 4.0 * 5.0 (Const) = 20.0 degrees
    rotor_drift = DreamRotor("Drifting Idea", [0.0]*21, void_distance=4.0)
    
    logger.info(f"Drift Rotor Tilt: {rotor_drift.tilt_angle}")
    if rotor_drift.tilt_angle > 19.0:
        logger.info("PASS: Distance converted to Tilt.")
    else:
        logger.error("FAIL: Tilt calculation incorrect.")
        return False
        
    # Test 3: Collapse Event
    # Distance 7.0 (Limit 6.0) -> Should Collapse
    rotor_collapse = DreamRotor("Impossible Thought", [0.0]*21, void_distance=7.0)
    status = rotor_collapse.check_integrity()
    
    logger.info(f"Collapse Status: {status['status']}")
    if status['status'] == "COLLAPSE":
        logger.info("PASS: Rotor collapsed due to Void Tether Snapback.")
    else:
        logger.error(f"FAIL: Rotor survived event horizon. Status: {status['status']}")
        return False
        
    return True

def test_protocol_integration():
    logger.info("\n[TEST] DreamProtocol Integration (Physics Context)")
    
    alchemist = DreamAlchemist()
    alchemist.cortex = MagicMock()
    alchemist.cortex.is_active = True
    alchemist.cortex.think = MagicMock(return_value="Physics Observed")
    
    # Mock Hardware (Painful but Clear)
    class MockMonitor:
        def sense_vitality(self):
            return {
                "cpu": BioSignal("Test", 0.8, "Pain"), # Should Trigger ~8200 RPM
                "ram": BioSignal("Test", 0.0, "Clear")
            }
    alchemist.monitor = MockMonitor()

    # Inject Dream
    # Use distance 2.0 (Tilt ~10 deg)
    dream_item = {
        "intent": "Spinning Top",
        "vector_dna": [0.0]*21,
        "tether_status": {"distance": 2.0}
    }
    
    with open(alchemist.queue_path, "w") as f:
        json.dump([dream_item], f)
        
    alchemist.sleep()
    
    calls = alchemist.cortex.think.call_args_list
    prompt = calls[0][0][0]
    
    # Check for Physics Report in Prompt
    if "ROTOR SPEED" in prompt and "AXIS TILT" in prompt:
        logger.info("PASS: Dream Prompt contains Physics Data.")
        if "8200 RPM" in prompt or "SPIN_OUT" in prompt: # Depending on logic
             pass
    else:
        logger.error("FAIL: Physics Data missing from prompt.")
        return False
        
    return True

if __name__ == "__main__":
    if test_rotor_physics() and test_protocol_integration():
        print("\nALL MERKABA PHYSICS VERIFIED.")
    else:
        print("\nVERIFICATION FAILED.")
