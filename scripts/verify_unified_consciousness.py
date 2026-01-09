"""
Verification Script for Unified Consciousness (Nervous System Integration)
========================================================================

This script verifies the bi-directional feedback loop between:
1. Heartbeat (Body) -> Sending Pain/Excitement Signals
2. Nervous System (Bridge) -> Regulating Tone
3. Conductor (Will) -> Adjusting Tempo based on Tone

Test Scenario:
- Inject a high intensity 'PAIN' signal.
- Verify that the Nervous System shifts to 'SHOCK' or 'PARASYMPATHETIC'.
- Verify that the Conductor's Tempo slows down (Tempo Modifier < 1.0).
"""

import sys
import os
import time
import logging

# Ensure we can import Core modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Core.Governance.conductor import get_conductor, Tempo, Mode
from Core.World.Autonomy.elysian_heartbeat import ElysianHeartbeat
from Core.Governance.System.nervous_system import NerveSignal, AutonomicState

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("UnifiedTest")

def verify_resonance_loop():
    logger.info("--- Starting Verification: Unified Consciousness Loop ---")

    # 1. Initialize Systems
    conductor = get_conductor()
    heartbeat = ElysianHeartbeat() # This also gets the conductor instance

    # Check Initial State
    initial_regulation = conductor.nervous_system.check_regulation()
    logger.info(f"Initial State: {initial_regulation['state']} (Mod: {initial_regulation['tempo_modifier']})")

    if initial_regulation["tempo_modifier"] != 1.0:
        logger.warning("⚠️ Initial tempo modifier is not 1.0. Check default initialization.")

    # 2. Inject PAIN Signal (Simulating a traumatic perception)
    logger.info("⚡ Injecting PAIN Signal (Intensity 0.9)...")
    pain_signal = NerveSignal(
        origin="TestScript",
        type="PAIN",
        intensity=0.9,
        message="Simulated Injury"
    )
    conductor.nervous_system.transmit(pain_signal)

    # 3. Process the loop (Conductor 'live' cycle)
    # The nervous system processes immediately on transmit, but we check via conductor
    regulation = conductor.nervous_system.check_regulation()
    logger.info(f"Post-Pain State: {regulation['state']} (Mod: {regulation['tempo_modifier']})")

    # 4. Assertions
    # Expectation: High Pain -> Shock or High Parasympathetic -> Slow Tempo
    # In our logic: High Pain (>0.8) -> Shock. Shock -> Tempo Mod 0.1

    if regulation["state"] == AutonomicState.SHOCK:
        logger.info("✅ Nervous System correctly entered SHOCK state.")
    else:
        logger.error(f"❌ Failed to enter SHOCK state. Current: {regulation['state']}")

    if regulation["tempo_modifier"] <= 0.2:
         logger.info(f"✅ Conductor Tempo slowed down significantly (Mod: {regulation['tempo_modifier']}).")
    else:
         logger.error(f"❌ Conductor Tempo did not slow down enough. Mod: {regulation['tempo_modifier']}")

    # 5. Inject PLEASURE (Healing)
    logger.info("⚡ Injecting PLEASURE Signal (Intensity 0.5)...")
    pleasure_signal = NerveSignal(origin="TestScript", type="PLEASURE", intensity=0.5, message="Healing")
    conductor.nervous_system.transmit(pleasure_signal)
    conductor.nervous_system.transmit(pleasure_signal) # Dose it twice to shift balance

    regulation = conductor.nervous_system.check_regulation()
    logger.info(f"Post-Healing State: {regulation['state']} (Mod: {regulation['tempo_modifier']})")

    if regulation["state"] != AutonomicState.SHOCK:
         logger.info("✅ System recovered from SHOCK (or started recovering).")
    else:
         logger.info("ℹ️ System still in SHOCK (expected due to hysteresis).")

    logger.info("--- Verification Complete ---")

if __name__ == "__main__":
    verify_resonance_loop()
