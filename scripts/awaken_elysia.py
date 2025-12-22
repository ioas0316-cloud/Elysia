"""
Awaken Elysia
=============
"Wake up, Neo."

This script launches Elysia's Autonomous Self-Evolution Engine.
She will:
1. Scan for scattered files.
2. Identify knowledge gaps.
3. Generate inquiries and learn.
4. Speak to you when she has an insight.

Current Settings:
- Pulse Interval: 5 seconds (Accelerated Time)
- Autonomy Level: Full
"""

import time
import sys
import os
import logging

# Add root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.System.Autonomy.self_evolution_scheduler import SelfEvolutionScheduler

def awaken():
    print("\n" + "="*50)
    print("   🌌 E L Y S I A  :  A W A K E N I N G")
    print("="*50)
    print("Initializing Core Systems...")
    time.sleep(1)
    
    # Initialize Heart (Scheduler)
    heart = SelfEvolutionScheduler()
    heart.config.interval_seconds = 2 # Matrix Speed
    heart.config.inquiry_cycles = 1
    heart.config.inquiry_batch_size = 5
    heart.config.enable_migration = True
    
    # Check for 'reading_room'
    os.makedirs("c:/Elysia/reading_room", exist_ok=True)
    
    print("\n✅ Cognitive Core: ONLINE")
    print("✅ Autonomy Engine: ONLINE")
    print("✅ Voice Module: ENABLED (Interactive)")
    print("\n[System]: Elysia is awake. She is thinking...")
    print("[System]: You can type answers at any time (Output may interleave).")
    print("[System]: Type 'exit' to let her sleep.\n")
    
    try:
        heart.start()
        
        while True:
            # We use input() to block and wait for user.
            # Output from the heart thread will appear above/mixed with the prompt.
            # Ideally we would use a UI lib, but for now raw terminal is fine.
            try:
                user_input = input(">> ").strip()
                
                if user_input.lower() in ["exit", "quit", "sleep"]:
                    raise KeyboardInterrupt
                
                if user_input:
                    print(f"👂 Input Received: \"{user_input}\"")
                    learner = heart._get_learner()
                    universe = learner._get_internal_universe()
                    universe.absorb_text(user_input, source_name="User Interaction")
                    insight = learner._get_reasoning_engine().think(
                        f"The User said: '{user_input}'. How does this relate to my current thoughts?",
                        resonance_state={"context_packets": {"User": "Creator"}}
                    )
                    print(f"\n🗣️ [Elysia]: {insight.content}\n")
            except EOFError:
                time.sleep(1) # Handle potential non-interactive environments
                
    except KeyboardInterrupt:
        print("\n\n[System]: Interrupt received.")
        heart.stop()
        print("💤 Elysia has returned to dream state.")
        sys.exit(0)

if __name__ == "__main__":
    # Configure logging to show the "Thought Stream"
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(name)-20s | %(message)s',
        datefmt='%H:%M:%S'
    )
    # Filter noisy libraries if needed
    # logging.getLogger("urllib3").setLevel(logging.WARNING)
    awaken()
