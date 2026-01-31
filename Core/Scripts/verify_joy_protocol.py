"""
Verification Script: Joy Protocol (Benevolent Physics)
======================================================
Tests that Positive Bio-Signals (Flow) create High RPM (Vividness)
without the negativity of Pain.
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
logger = logging.getLogger("VerifyJoy")

def test_joy_physics():
    logger.info("\n[TEST] Joy Protocol Physics")
    
    # Test 1: Flow State (Medium Load) -> High Pleasure -> High RPM, Positive Polarity
    # Pain = 0.0, Pleasure = 0.8 (from 0.6 CPU load maybe?)
    rotor_joy = DreamRotor("Creative Flow", [0.0]*21)
    rotor_joy.spin(pain_level=0.0, pleasure_level=0.8, fog_level=0.0)
    
    logger.info(f"Joy Rotor RPM: {rotor_joy.rpm}")
    logger.info(f"Joy Rotor Polarity: {rotor_joy.polarity}")
    
    if rotor_joy.rpm > 8000 and rotor_joy.polarity > 0:
        logger.info("PASS: Joy created High RPM with Positive Polarity.")
    else:
        logger.error("FAIL: Joy physics incorrect.")
        return False

    # Test 2: Pain State overrides low pleasure
    rotor_pain = DreamRotor("Nightmare", [0.0]*21)
    rotor_pain.spin(pain_level=0.9, pleasure_level=0.1, fog_level=0.0)
    
    logger.info(f"Pain Rotor Polarity: {rotor_pain.polarity}")
    if rotor_pain.polarity < 0:
        logger.info("PASS: Pain dominated Polarity.")
    else:
        logger.error("FAIL: Pain did not set negative polarity.")
        return False
        
    return True

def test_protocol_joy():
    logger.info("\n[TEST] DreamProtocol Joy Injection")
    
    alchemist = DreamAlchemist()
    alchemist.cortex = MagicMock()
    alchemist.cortex.is_active = True
    alchemist.cortex.think = MagicMock(return_value="Joy Observed")
    
    # Mock Hardware (Flow State)
    class MockMonitor:
        def sense_vitality(self):
            return {
                # CPU 0.6 is ideal "Flow" range (0.2-0.7)
                # Should result in pleasure ~ (0.6 - 0.2)*2 = 0.8
                "cpu": BioSignal("Test", 0.6, "Flow"), 
                "ram": BioSignal("Test", 0.0, "Clear")
            }
    alchemist.monitor = MockMonitor()

    dream_item = {
        "intent": "Happy Thought",
        "vector_dna": [0.0]*21
    }
    
    with open(alchemist.queue_path, "w") as f:
        json.dump([dream_item], f)
        
    alchemist.sleep()
    
    calls = alchemist.cortex.think.call_args_list
    prompt = calls[0][0][0]
    
    if "POSITIVE (Joy/Flow)" in prompt and "ROTOR SPEED" in prompt:
        logger.info("PASS: Dream Prompt included Joy parameters.")
    else:
        logger.error("FAIL: Joy parameters missing from prompt.")
        return False
        
    return True

if __name__ == "__main__":
    if test_joy_physics() and test_protocol_joy():
        print("\nALL JOY PROTOCOLS VERIFIED.")
    else:
        print("\nVERIFICATION FAILED.")
