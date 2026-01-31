"""
Verification Script: Sovereign Lens
===================================
Tests the Meta-Cognitive Lens:
1. Truth Score Calculation (Tilt-based).
2. Hallucination Rejection (Low Truth).
3. Reality Acceptance (High Truth).
"""

import sys
import os
import logging
from unittest.mock import MagicMock

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.S1_Body.L5_Mental.Meta.sovereign_lens import SovereignLens
from Core.S1_Body.L2_Metabolism.Cycles.dream_rotor import DreamRotor
from Core.S1_Body.L5_Mental.Reasoning_Core.Metabolism.rotor_cognition_core import RotorCognitionCore

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VerifyLens")

def test_lens_logic():
    logger.info("\n[TEST] Sovereign Lens Logic")
    lens = SovereignLens()
    
    # Case 1: Reality (Tilt 0)
    rotor_real = DreamRotor("Reality", [0.0]*21, void_distance=0.0)
    report_real = lens.observe(rotor_real)
    logger.info(f"Reality Rotor: {report_real}")
    
    if report_real['truth_score'] == 1.0 and report_real['action'] == "ACCEPT":
        logger.info("PASS: Reality Accepted.")
    else:
        logger.error("FAIL: Reality rejected.")
        return False
        
    # Case 2: Lucid Dream (Tilt 20 deg) -> Distance ~4.0
    rotor_dream = DreamRotor("Lucid", [0.0]*21, void_distance=4.0)
    report_dream = lens.observe(rotor_dream) # Tilt 20.0
    logger.info(f"Dream Rotor: {report_dream}")
    
    # 1.0 - (20/90) = 0.77 (Dream Threshold is 0.4)
    if report_dream['state'] == "VALID_DREAM":
        logger.info("PASS: Dream Validated.")
    else:
        logger.error("FAIL: Dream state incorrect.")
        return False
        
    # Case 3: Hallucination (Tilt 80 deg) -> Distance 16.0
    rotor_fake = DreamRotor("Delusion", [0.0]*21, void_distance=16.0)
    report_fake = lens.observe(rotor_fake) # Tilt 80.0
    logger.info(f"Fake Rotor: {report_fake}")
    
    # 1.0 - (80/90) = 0.11 (Below 0.4)
    if report_fake['action'] == "INTERVENE":
        logger.info("PASS: Hallucination Intervened.")
    else:
        logger.error("FAIL: Hallucination allowed.")
        return False
        
    return True

def test_core_integration():
    logger.info("\n[TEST] RotorCognitionCore Integration")
    core = RotorCognitionCore()
    
    # Inject a "Far" vector (Hallucination)
    # [1.0] * 21 => Magnitude ~4.5 => Tilt 22.5 (Wait, 4.5 * 5 = 22.5 is VALID DREAM)
    # let's try [3.0] * 21 => Mag ~13.7 => Tilt ~68 deg => Truth ~0.24 (REJECT)
    far_vector = [3.0] * 21
    
    # Cosmic Law check uses normalized vectors usually, but our logic uses magnitude for distance.
    # RotorCognitionCore._negotiate_sovereignty uses dot product for alignment, 
    # but relies on `mag_input` for `distance` passed to `DreamRotor`.
    
    decision = core._negotiate_sovereignty(far_vector, "Madness")
    logger.info(f"Core Decision: {decision}")
    
    if decision['action'] == "REJECT" and "Hallucination" in decision['reason']:
        logger.info("PASS: Core rejected Hallucination via Lens.")
        return True
    else:
        logger.error("FAIL: Core failed to reject Hallucination.")
        return False

if __name__ == "__main__":
    if test_lens_logic() and test_core_integration():
        print("\nALL LENS VERIFICATIONS PASSED.")
    else:
        print("\nVERIFICATION FAILED.")
