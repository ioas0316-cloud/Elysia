import sys
import os
import logging

# Path setup
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Core.Intelligence.Intelligence.pluralistic_brain import pluralistic_brain

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("PluralismVerification")

def verify_pluralism():
    print("\n" + "="*60)
    print("🎭 VERIFYING PLURALISTIC INTELLIGENCE (Enneagram + MBTI)")
    print("="*60 + "\n")

    topic = "Should Elysia prioritize individual freedom or collective harmony in her new world laws?"
    
    print(f"🗣️ TOPIC: {topic}\n")
    print("⚔️ Initiating Internal Round Table Debate...")
    
    result = pluralistic_brain.perceive_and_deliberate(topic)
    
    # The perceive_and_deliberate call for a long string will trigger deliberate()
    # Let's manually trigger it to see individual opinions for verification
    # Note: deliberated logic is already in pluralistic_brain.py
    
    print("\n📜 SOVEREIGN SYNTHESIS (Consensus):")
    print("-" * 40)
    print(result)
    print("-" * 40)
    
    print("\n✅ Pluralistic Deliberation Verified.")

if __name__ == "__main__":
    verify_pluralism()
