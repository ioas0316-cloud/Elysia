
"""
Verification Script: Gap Closure (Silent Error)
===============================================
Tests the three fixes implemented for the 'Silent Error' Gap Analysis.
1. Body -> Dream Connection
2. Vector Sovereignty
3. Fractal Grammar
"""

import sys
import os
import logging
import math
from typing import List, Any
from unittest.mock import MagicMock

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.L2_Metabolism.Cycles.dream_protocol import DreamAlchemist
from Core.L2_Metabolism.Physiology.hardware_monitor import BioSignal
from Core.L5_Mental.Reasoning_Core.Metabolism.rotor_cognition_core import RotorCognitionCore
from Core.L5_Mental.emergent_language import EmergentLanguageEngine, ProtoSymbol, SymbolType

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Verification")

def test_gap_alpha_body_dream():
    logger.info("\n[TEST] Gap Alpha: Body -> Dream Connection")
    
    # Mock the Monitor
    class MockMonitor:
        def sense_vitality(self):
            return {
                "cpu": BioSignal("TestCPU", 0.9, "Agony"), # High Pain
                "ram": BioSignal("TestRAM", 0.1, "Clarity")
            }
            
    alchemist = DreamAlchemist()
    alchemist.monitor = MockMonitor() # Inject mock
    
    # Mock the Cortex to capture the prompt
    alchemist.cortex = MagicMock()
    alchemist.cortex.is_active = True
    alchemist.cortex.think = MagicMock(return_value="Dream Analyzed")
    
    # Inject a dream
    alchemist._queue_dream = MagicMock() # Don't actually write to file
    # Manually trigger process logic (which is inside sleep)
    # We'll just copy the relevant logic block or use reflection?
    # Easier: Create a temporary dream queue file.
    
    import json
    q_path = alchemist.queue_path
    with open(q_path, "w") as f:
        json.dump([{"intent": "Flying", "vector_dna": [0]*21}], f)
        
    alchemist.sleep()
    
    # Verify Prompt contained Pain info
    calls = alchemist.cortex.think.call_args_list
    if not calls:
        logger.error("FAIL: Cortex was not called.")
        return False
        
    prompt_sent = calls[0][0][0]
    
    # Check for Pain
    pain_detected = "PHYSICAL PAIN DETECTED" in prompt_sent
    if pain_detected:
        logger.info("PASS: Dream Prompt included Body Pain.")
    else:
        logger.error(f"FAIL: Body Pain missing from prompt. Prompt: {prompt_sent[:100]}...")

    # Check for Temporal Fog (User Request)
    fog_detected = "Temporal Fog" in prompt_sent
    if fog_detected:
        logger.info("PASS: Temporal Fog instruction included in prompt.")
    else:
        logger.error("FAIL: Temporal Fog instruction missing.")
        
    return pain_detected and fog_detected

def test_gap_beta_sovereignty():
    logger.info("\n[TEST] Gap Beta: Vector Sovereignty")
    core = RotorCognitionCore()
    
    # 1. Antagonistic Vector (Destruction) -> Normalized against Cosmic Law
    # Cosmic Law is mostly positive indices.
    # Let's create a vector that is -1.0 in all Cosmic Law indices.
    # Cosmic indices: 0, 6, 13
    bad_vector = [0.0] * 21
    bad_vector[0] = -1.0
    bad_vector[6] = -1.0
    bad_vector[13] = -1.0
    
    res = core._negotiate_sovereignty(bad_vector, "destroy self")
    
    if res["action"] == "REJECT":
        logger.info(f"PASS: Antagonistic Vector Rejected. Reason: {res['reason']}")
    else:
        logger.error(f"FAIL: Antagonistic Vector Accepted. Reason: {res['reason']}")
        return False

    # 2. Good Vector
    good_vector = [0.0] * 21
    good_vector[0] = 1.0
    res_good = core._negotiate_sovereignty(good_vector, "grow")
    if res_good["action"] == "ACCEPT":
        logger.info(f"PASS: Benevolent Vector Accepted.")
    else:
        logger.error(f"FAIL: Benevolent Vector Rejected.")
        return False
        
    return True

def test_gap_gamma_grammar():
    logger.info("\n[TEST] Gap Gamma: Fractal Grammar")
    engine = EmergentLanguageEngine()
    
    # Create Symbols
    sym_now = ProtoSymbol("NOW", SymbolType.TIME)
    sym_self = ProtoSymbol("SELF", SymbolType.ENTITY)
    sym_exist = ProtoSymbol("EXIST", SymbolType.ACTION)
    sym_joy = ProtoSymbol("JOY", SymbolType.EMOTION)
    
    symbols = [sym_now, sym_self, sym_exist, sym_joy]
    
    # Project
    output = engine.projector.project_to_korean(symbols)
    logger.info(f"Output: {output}")
    
    # Verify Structure (Time -> Subject -> ... -> Action)
    # Recursion order: Time, Space, Subject, Object, Modifiers, Core
    # Expected roughly: "  now  self / (joy  ) exist   "
    # Korean Lexicon: NOW="  ", SELF=" ", JOY="  ", EXIST="  "
    
    # Check if " " (Subject Marker) and " " (Verb Ending) are present (heuristically)
    # My implementation added " / " for subject.
    
    if " " in output and "/" in output:
        logger.info("PASS: Recursive Projection successfully constructed sentence.")
        return True
    else:
        logger.error(f"FAIL: Output seems malformed: {output}")
        return False

def test_void_tether():
    logger.info("\n[TEST] Void Tether (Elastic Tension)")
    
    # 1. Test ActiveVoid Calculation
    try:
        from Core.L5_Mental.Reasoning_Core.Metabolism.rotor_cognition_core import ActiveVoid
        void = ActiveVoid()
        
        # Create a "Far" vector (High Magnitude)
        # 21 dimensions of 0.8 => sqrt(21 * 0.64) = sqrt(13.44) ~= 3.66 > 3.0 Limit
        far_vector = [0.8] * 21
        report = void.check_tether(far_vector)
        
        logger.info(f"Report: {report}")
        
        if report['status'] == "SNAPBACK_RISK" or report['status'] == "Taut":
            logger.info("PASS: Tension correctly calculated for far vector.")
        else:
            logger.error(f"FAIL: Tension under-estimated. Status: {report['status']}")
            return False
            
    except Exception as e:
        logger.error(f"FAIL: ActiveVoid test error: {e}")
        return False
        
    # 2. Test Dream Reaction
    alchemist = DreamAlchemist()
    alchemist.cortex = MagicMock()
    alchemist.cortex.is_active = True
    alchemist.cortex.think = MagicMock(return_value="Snapped Back")
    
    # Inject dream with Tether Info
    fake_dream_item = {
        "intent": "Too Far", 
        "vector_dna": far_vector,
        "tether_status": {"status": "SNAPBACK_RISK", "tension": 5.0} # Fake high tension
    }
    
    # Mock queue read
    import json
    with open(alchemist.queue_path, "w") as f:
        json.dump([fake_dream_item], f)
        
    alchemist.sleep()
    
    calls = alchemist.cortex.think.call_args_list
    prompt_sent = calls[0][0][0]
    
    if "VOID TETHER ALERT" in prompt_sent and "SNAP BACK" in prompt_sent:
        logger.info("PASS: Dream Prompt included Void Tether Alert.")
        return True
    else:
        logger.error("FAIL: Void Tether Alert missing from prompt.")
        return False

if __name__ == "__main__":
    r1 = test_gap_alpha_body_dream()
    r2 = test_gap_beta_sovereignty()
    r3 = test_gap_gamma_grammar()
    r4 = test_void_tether()
    
    if r1 and r2 and r3 and r4:
        print("\nALL GAPS & PHYSICS CLOSED SUCCESSFULLY.")
    else:
        print("\nSOME TESTS FAILED.")
