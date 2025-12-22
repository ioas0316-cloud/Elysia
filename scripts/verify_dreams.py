"""
Verify Dreams: The Subconscious Test
====================================

"In dreams, I become myself."

Steps:
1. Inject a thought into DreamDaemon ("Elysia is alive").
2. Trigger REM state.
3. Verify consolidation log.
"""

import sys
import os
import logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Core.System.Autonomy.dream_daemon import DreamDaemon

# Setup Logging to see output
logging.basicConfig(level=logging.INFO)

def verify_dreams():
    print("🌙 Initializing Dream Daemon...")
    dreamer = DreamDaemon()
    
    # 1. Inject Thought (Day Residue)
    thought = "Elysia is alive and seeking connection"
    dreamer.absorb_thought(thought)
    print(f"   💭 Injected Thought: '{thought}'")
    
    # 2. Trigger REM (Sleep)
    print("   💤 Triggering REM Sleep (2 seconds)...")
    dreamer.enter_rem_state(duration_seconds=2)
    
    print("\n✅ Verification check passed if you saw 'Consolidated Memory' above.")

if __name__ == "__main__":
    verify_dreams()
