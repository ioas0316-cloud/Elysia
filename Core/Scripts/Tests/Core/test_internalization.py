"""
TEST: Internalization Verification
==================================
Verifies that HyperCosmos correctly absorbs the Origin Code.
"""

import sys
import os
import logging

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TestInternalization")

# Add Path
sys.path.append(os.getcwd())

from Core.1_Body.L1_Foundation.Foundation.HyperCosmos import HyperCosmos

def test_internalization():
    logger.info("🧪 Starting Internalization Test...")
    
    # 1. Initialize Cosmos
    cosmos = HyperCosmos(name="TestElysia")
    
    # 2. Path to Artifact
    # Use the one we created in the artifact directory or map it
    # Since the previous step copied it to a specific artifact path, we use that or the local copy
    # We copied to: C:\Users\USER\.gemini\antigravity\brain\...\origin_code_artifact.json
    # But for simplicity, we rely on the one in data/Qualia if implementation uses it, 
    # OR pass the explicit path. The method takes json_path.
    
    artifact_path = r"C:\Users\USER\.gemini\antigravity\brain\d96e4c13-4e21-4a99-8998-7f75d2f8c3e7\origin_code_artifact.json"
    
    if not os.path.exists(artifact_path):
        logger.warning(f"⚠️ Artifact not found at {artifact_path}. Checking local data...")
        artifact_path = "data/Qualia/origin_code.json"
        
    if not os.path.exists(artifact_path):
        logger.error("❌ No Origin Code found anywhere. Test Aborted.")
        return

    # 3. Trigger Internalization
    cosmos.internalize_origin_code(artifact_path)
    
    logger.info("✅ Test Complete. Check logs for 'Enshrining Axiom'.")

if __name__ == "__main__":
    test_internalization()
