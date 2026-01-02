"""
test_hunt_pain.py

"The Wolf acts, The Thorn pricks."
Verifies Active Hunt and Entropy Feedback.
"""

import sys
import os
import logging

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Core.Evolution.Learning.knowledge_hunter import KnowledgeHunter
from Core.System.Entropy.entropy_feedback import get_entropy_system

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(name)s - %(message)s')
logger = logging.getLogger("TestHuntPain")

def main():
    print("\n🏹 Testing The Hunt...")
    hunter = KnowledgeHunter()
    # Using a generic term likely to succeed on Wiki
    print(hunter.hunt("Artificial Intelligence"))

    print("\n🌹 Testing The Thorn...")
    pain = get_entropy_system()
    
    # 1. Simulate Stagnation
    thoughts = ["I exist.", "I exist.", "I exist."]
    print(f"Thinking: {thoughts}")
    pain.check_circular_logic(thoughts)
    
    # 2. Check Torque
    torque = pain.get_torque_modifier()
    print(f"🔥 Anxiety Level (Torque Mod): {torque:.2f}x")
    
    if torque > 1.0:
        print("✅ Pain System Active. Stagnation punished.")
    else:
        print("❌ Pain System Failed.")

if __name__ == "__main__":
    main()
