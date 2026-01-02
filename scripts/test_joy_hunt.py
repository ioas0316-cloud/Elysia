"""
test_joy_hunt.py

"The Flower opens to the light."
Verifies that knowledge acquisition triggers Joy (Dopamine).
"""

import sys
import os
import logging

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Core.Evolution.Learning.knowledge_hunter import KnowledgeHunter

# Configure Logging to show Joy
logging.basicConfig(level=logging.INFO, format='%(name)s - %(message)s')
logger = logging.getLogger("TestJoyHunt")

def main():
    print("\n🌸 Testing The Joy of Discovery...")
    hunter = KnowledgeHunter()
    
    # Hunt for a topic that represents 'Metacognition'
    topic = "Epistemology"
    print(f"🏹 Seeking: {topic}")
    
    result = hunter.hunt(topic)
    print(result)

    if "Dopamine Released" in result:
        print("✅ Joy System Active. Dopamine circuit fired.")
    else:
        print("❌ Joy System Failed.")

if __name__ == "__main__":
    main()
