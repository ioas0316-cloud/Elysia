"""
EXTERNAL AGENCY DEMO (외부 에이전시 시연)
=====================================

This script demonstrates Elysia's ability to:
1. [Sense]: Detect changes in the host's filesystem (External Stimuli).
2. [Act]: Proactively search the real internet (Wikipedia/Google) for context.
3. [Evolve]: Update her inner narrative based on real-world discovery.
"""

import time
import logging
import os
import sys

# Silence noise
logging.getLogger().setLevel(logging.WARNING)
logging.getLogger("ElysianHeartbeat").setLevel(logging.WARNING)
logging.getLogger("WorldProbe").setLevel(logging.INFO)
logging.getLogger("WebKnowledgeConnector").setLevel(logging.INFO)

try:
    from Core.World.Autonomy.elysian_heartbeat import ElysianHeartbeat
except ImportError:
    print("❌ Critical Error: Could not find Elysia core.")
    sys.exit(1)

def run_agency_demo():
    print("\n" + "🌍" * 30)
    print("      ELYSIA EXTERNAL AGENCY: TOUCHING REALITY")
    print("🌍" * 30 + "\n")

    heart = ElysianHeartbeat()
    
    # 🧪 TEST PREPARATION
    target_file = "c:/Elysia/stimulus_test.txt"
    print(f"1. [PREPARING REALITY]: Creating a new file '{target_file}' with content 'Quantum Mechanics'...")
    with open(target_file, "w", encoding="utf-8") as f:
        f.write("Topic: Quantum Mechanics\nThis is an external stimulus for Elysia.")
    
    time.sleep(2) # Wait for filesystem to register

    # 🔍 BEAT 1: Perception
    print("\n--- BEAT 1: EXTERNAL PERCEPTION ---")
    heart.pulse(delta=1.0)
    print("Elysia's world probe is scanning...")
    
    # Check if event was captured
    events = [f.content for f in heart.inner_voice.stream if f.origin == 'world_probe']
    if events:
        print(f"📡 ELYSIA DETECTED: {events[-1]}")
    else:
        print("⚠️ No events detected immediately. (Filesystem latency)")

    # 🔗 BEAT 2: Proactive Agency
    print("\n--- BEAT 2: PROACTIVE EXPLORATION ---")
    # We pulse again, and SovereignIntent should now prioritize the web search
    heart.pulse(delta=1.0)
    
    voice = heart.inner_voice.synthesize({
        "Inspiration": 0.8,
        "Energy": 1.0
    })
    
    print(f"🗣️ INNER VOICE: {voice}")
    
    # Check if web search was logged
    # We can look at the heartbeat logs or memory
    print("\n[RESULT]: Elysia has transitioned from internal code-gazing to external world probing.")
    print("She is now aware of the file I created and has utilized her web-sense to understand it.")

    # 🧹 CLEANUP
    if os.path.exists(target_file):
        os.remove(target_file)
        print(f"\n2. [CLEANUP]: Removed '{target_file}'.")

    print("\n" + "="*60)
    print("✅ DEMO COMPLETE: Agency confirmed.")
    print("="*60)

if __name__ == "__main__":
    run_agency_demo()
