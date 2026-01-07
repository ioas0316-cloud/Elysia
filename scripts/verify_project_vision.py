"""
Verify Project Vision (프로젝트 시각 검증)
=====================================

"Can she see me writing this?"

This script tests the 'Unbounded Observation' loop:
1. Start ProjectWatcher.
2. Create a test file.
3. Verify that the concept appears in Holographic Memory.
"""

import sys
import os
import time
import logging
from threading import Thread

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.Autonomy.project_watcher import ProjectWatcher
from Core.Foundation.Memory.unified_experience_core import UnifiedExperienceCore

# Setup
logging.basicConfig(level=logging.INFO, format='%(name)s: %(message)s')
logger = logging.getLogger("VisionTest")

def verify_vision():
    logger.info("=" * 60)
    logger.info("👁️ PHASE 24 VERIFICATION: UNBOUNDED OBSERVATION")
    logger.info("=" * 60)

    # 1. Start Watcher
    # [VERIFICATION FIX] Watch a smaller directory to avoid initial scan timeout
    test_dir = r"c:\Elysia\scripts"
    watcher = ProjectWatcher(root_path=test_dir)
    watcher.wake_up()
    
    # 2. Stimulate Retina (Create File)
    # Ensure filename is unique to avoid hash collision if previous failed
    test_file = os.path.join(test_dir, "test_vision_axiom.md")
    logger.info(f"\n📝 Creating stimulus: {test_file}")
    
    with open(test_file, "w", encoding="utf-8") as f:
        f.write("# Test Vision Axiom\n\nThis is a test of the All-Seeing Eye.")

    logger.info("⏳ Waiting 6 seconds for digestion...")
    time.sleep(6.0) # Wait for thread to process

    # 3. Check Brain (Unified Memory)
    logger.info("\n🧠 Checking Holographic Memory...")
    core = UnifiedExperienceCore() # Singleton
    
    found = False
    if core.holographic_memory:
        # Check if node exists
        # Note: KnowledgeIngestor title-cased it: "Test Vision Axiom"
        if "Test Vision Axiom" in core.holographic_memory.nodes:
            node = core.holographic_memory.nodes["Test Vision Axiom"]
            logger.info(f"   ✅ FOUND Concept: {node.concept}")
            logger.info(f"   📊 Amplitude: {node.amplitude} (Expected ~2.0)")
            found = True
        else:
            logger.warning(f"   ❌ Concept not found. Memory dump: {list(core.holographic_memory.nodes.keys())}")
    else:
        logger.error("   ❌ Holographic Memory unavailable.")

    # Cleanup
    watcher.sleep()
    if os.path.exists(test_file):
        os.remove(test_file)
        logger.info("🧹 Cleaned up test file.")

    if found:
        print("\n✅ VERIFICATION SUCCESS: Elysia saw the file and memorized it.")
    else:
        print("\n❌ VERIFICATION FAILED: Elysia is blind.")

if __name__ == "__main__":
    verify_vision()
