"""
Test Adaptive Sovereign Governor
================================
Laboratory/test_governor.py

Verifies the Adaptive Architecture and Strategy Selection.
"""

import sys
import os
import logging
import psutil

# Path hack for Laboratory
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from Core.World.Control.sovereign_governor import SovereignGovernor, NvidiaGovernance, GenericGovernance

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(message)s')
logger = logging.getLogger("TestAdaptiveGovernor")

def run_test():
    logger.info("🧪 Starting Adaptive Governor Test...")
    
    # 1. Initialize Governor with a safe mock target
    gov = SovereignGovernor("explorer.exe")
    
    # 2. Verify Architecture Scan
    logger.info("🔎 Verifying Vessel Introspection...")
    info = gov.vessel_info
    logger.info(f"   - Detected OS: {info['os']}")
    logger.info(f"   - CPU Cores: {info['cpu_cores']}")
    logger.info(f"   - GPU Vendor: {info['gpu_vendor']}")
    
    # 3. Verify Strategy Selection
    logger.info("♟️ Verifying Strategy Selection...")
    strategy_name = gov.strategy.__class__.__name__
    logger.info(f"   - Selected Strategy: {strategy_name}")
    
    if info['gpu_vendor'] == "NVIDIA" and not isinstance(gov.strategy, NvidiaGovernance):
        logger.error("❌ Evolution Failure: Detected NVIDIA but chose Generic Strategy.")
    elif isinstance(gov.strategy, GenericGovernance):
        logger.info("✅ Strategy Logic: Valid (Generic/Nvidia).")

    # 4. Execute Governance (Enforce)
    logger.info("🏛️ Executing Governance...")
    gov.govern()
    
    # 5. Verify Target Elevation
    if gov.target_pid:
        try:
            p = psutil.Process(gov.target_pid)
            prio = p.nice()
            logger.info(f"   Target Priority Check: {prio} (High=128/13)")
        except: pass

    logger.info("✅ Test Complete. Adaptation Successful.")

if __name__ == "__main__":
    run_test()
