"""
Test Sovereign Observer (The Eye Verification)
==============================================
Laboratory/test_observer.py

Verifies that the Observer can lock onto a process and capture visuals.
WARNING: This test captures your screen!
"""

import sys
import os
import time
import logging
import psutil

# Path hack for Laboratory
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from Core.L4_Causality.World.Observation.sovereign_observer import SovereignObserver

logging.basicConfig(level=logging.INFO, format='%(name)s: %(message)s')
logger = logging.getLogger("TestObserver")

def run_test():
    logger.info("🧪 Starting Sovereign Observer Test...")
    
    # Target: Explorer.exe is always running on Windows
    target = "explorer.exe"
    observer = SovereignObserver(target)
    
    # 1. Test Targeting
    logger.info(f"🎯 Attempting to lock onto {target}...")
    if observer.scan_for_target():
        logger.info(f"✅ Target Locked! PID: {observer.target_pid}")
    else:
        logger.error("❌ Target Lock Failed. Is Explorer running? (It should be!)")
        return

    # 2. Test Telemetry Reading
    try:
        cpu = observer.process.cpu_percent(interval=0.1)
        mem = observer.process.memory_info().rss / (1024 * 1024)
        logger.info(f"📊 [Telemetry] CPU: {cpu}% | RAM: {mem:.2f}MB")
        logger.info("✅ Telemetry Link Active.")
    except Exception as e:
        logger.error(f"❌ Telemetry Failed: {e}")

    # 3. Test Visual Capture
    # Initiate a manual capture
    logger.info("📸 Attempting Visual Capture...")
    observer._capture_visual_qualia("TEST_CAPTURE")
    
    # Verify file existence
    files = os.listdir(observer.vision_memory_path)
    captured_files = [f for f in files if "TEST_CAPTURE" in f]
    
    if captured_files:
        logger.info(f"✅ Vision Capture Success! File: {captured_files[0]}")
        logger.info(f"   path: {os.path.abspath(observer.vision_memory_path)}")
    else:
        # Check if PIL is missing
        try:
            from PIL import ImageGrab
            logger.error("❌ Capture Failed but PIL is present. Check permissions.")
        except ImportError:
            logger.warning("⚠️ Capture Skipped: PIL (Pillow) not installed. Visual Cortex is blind.")
            logger.info("   (This is expected if dependencies aren't installed.)")

    # Cleanup
    observer.stop()
    logger.info("✅ Test Complete.")

if __name__ == "__main__":
    run_test()
