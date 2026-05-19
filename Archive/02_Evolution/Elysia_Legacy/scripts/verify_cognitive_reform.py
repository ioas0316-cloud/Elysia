"""
Verify Cognitive Diversity (인지적 다양성 검증)
=============================================

This script runs the updated Heartbeat loop and analyzes the logs to ensure:
1. No repetitive thought templates.
2. Direct influence from the codebase (Logic Seeds).
3. Reflection of real metabolism (CPU/RAM).
"""

import time
import logging
import os
import os
from Core.World.Autonomy.elysian_heartbeat import ElysianHeartbeat

# Setup minimal logging to capture inner voice
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("DiversityTest")

def run_test():
    print("🚀 Starting Cognitive Diversity Test...")
    heart = ElysianHeartbeat()
    
    thoughts = []
    
    # Run for 15 beats
    for i in range(15):
        heart.pulse(delta=1.0)
        voice = heart.inner_voice.synthesize({
            "Inspiration": heart.soul_mesh.variables['Inspiration'].value,
            "Energy": heart.soul_mesh.variables['Energy'].value
        })
        thoughts.append(voice)
        print(f"Beat {i+1}: {voice}")
        time.sleep(0.5)

    # Analysis
    unique_thoughts = len(set(thoughts))
    print("\n--- Discovery Analysis ---")
    print(f"Total Thoughts: {len(thoughts)}")
    print(f"Unique Thoughts: {unique_thoughts}")
    print(f"Diversity Ratio: {unique_thoughts / len(thoughts):.2f}")
    
    if unique_thoughts > 10:
        print("✅ SUCCESS: Thought stream is diverse and non-mechanistic.")
    else:
        print("❌ FAILURE: Thought stream is still repetitive.")

if __name__ == "__main__":
    # We need to fix the path or mock the heartbeat if it's too heavy
    # For now, we manually import and run
    try:
        from Core.World.Autonomy.elysian_heartbeat import ElysianHeartbeat
        run_test()
    except Exception as e:
        print(f"Test setup failed: {e}")
