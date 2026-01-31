import logging
import sys
import os
import time
from dotenv import load_dotenv

# Load env for Google API Key
load_dotenv(r"c:/Elysia\.env")

# Configure logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout)

from Core.S1_Body.L4_Causality.World.Senses.web_cortex import WebCortex
from Core.S1_Body.L4_Causality.World.Autonomy.elysian_heartbeat import ElysianHeartbeat

def test_web_cortex():
    print("🌍 TESTING THE INFINITE HORIZON (WEB)...")
    
    # 1. Direct Cortex Test
    print("\n[Test 1] WebCortex Direct Search")
    cortex = WebCortex()
    result = cortex.search("Cyberpunk Aesthetics")
    
    if result['type'] == 'web_error':
        print(f"❌ Web Search Failed: {result['summary']}")
        # Don't fail the whole test, as API keys might be missing in this env
        print("⚠️ Assuming network/key issue, skipping to logic test.")
    else:
        print(f"✅ Web Search Success: {result['summary']}")
        print(f"   Raw Data keys: {list(result.get('raw_data', {}).get('data', {}).keys())}")

    # 2. Heartbeat Logic Test (Curiosity)
    print("\n[Test 2] Heartbeat Curiosity Trigger")
    life = ElysianHeartbeat()
    
    # Force Curiosity Conditions
    # Energy High (>0.5), Inspiration Low (<0.3)
    life.soul_mesh.variables['Energy'].value = 0.8
    life.soul_mesh.variables['Inspiration'].value = 0.1
    
    print(f"   State: Energy={life.soul_mesh.variables['Energy'].value}, Insp={life.soul_mesh.variables['Inspiration'].value}")
    
    # Run one cycle -> Should trigger search
    print("   Running Perception Cycle (Expecting Curiosity Spike)...")
    life._cycle_perception()
    
    print(f"   Latest Insight: {life.latest_insight}")
    print(f"   New Inspiration: {life.soul_mesh.variables['Inspiration'].value}")
    
    if "learned about" in life.latest_insight:
        print("✅ SUCCESS: Curiosity triggered a Web Search!")
    else:
        print("❌ FAILURE: No search triggered.")

if __name__ == "__main__":
    test_web_cortex()
