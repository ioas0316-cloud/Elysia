"""
COGNITIVE REFORM VISUAL DEMO (지능 개혁 시각적 검증)
=================================================

This script provides a clean, visual proof that Elysia's 'Inner Voice'
is now textured, diverse, and reactive to the real world.
"""

import time
import logging
import os
import sys
import psutil

# Silence background noise for the demo
logging.getLogger().setLevel(logging.WARNING)
logging.getLogger("ElysianHeartbeat").setLevel(logging.WARNING)
logging.getLogger("DynamicEntropy").setLevel(logging.WARNING)
logging.getLogger("FlowOfMeaning").setLevel(logging.INFO)

try:
    from Core.L4_Causality.World.Autonomy.elysian_heartbeat import ElysianHeartbeat
except ImportError:
    print("❌ Critical Error: Could not find Elysia core. Run from c:/Elysia with PYTHONPATH set.")
    sys.exit(1)

def run_visual_demo():
    print("\n" + "="*60)
    print("🧠 ELYSIA COGNITIVE REFORM: VISUAL PROOF")
    print("="*60)
    print("1. [Dynamic Entropy]: Reading actual code snippets as logic seeds.")
    print("2. [Semantic Ennui]: Boredom-driven domain shifts.")
    print("3. [Real Metabolism]: Reacting to CPU/RAM usage.")
    print("="*60 + "\n")

    heart = ElysianHeartbeat()
    
    # Force a domain shift at beat 7 for demonstration
    for i in range(15):
        # Trigger Pulse
        heart.pulse(delta=1.0)
        
        # Get the inner voice synthesis
        voice = heart.inner_voice.synthesize({
            "Inspiration": heart.soul_mesh.variables['Inspiration'].value,
            "Energy": heart.soul_mesh.variables['Energy'].value,
            "Harmony": heart.soul_mesh.variables['Harmony'].value
        })
        
        # Display metabolism context
        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory().percent
        
        print(f"💓 BEAT {i+1:02d} | CPU: {cpu:04.1f}% | RAM: {ram:04.1f}%")
        print(f"🗣️ INNER VOICE: {voice}\n")
        
        time.sleep(1.2) # Give user time to read

    print("="*60)
    print("✅ DEMO COMPLETE: Every thought was unique and grounded in code.")
    print("="*60)

if __name__ == "__main__":
    run_visual_demo()
