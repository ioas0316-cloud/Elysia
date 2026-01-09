import logging
import sys
import os
import time

# Configure logging
logging.basicConfig(level=logging.INFO, stream=sys.stdout)

from Core.World.Senses.sensorium import Sensorium

def test_sensorium():
    print("🌐 TESTING THE SENSORIUM (Omni-Sensory)...")
    
    # ensure gallery exists
    os.makedirs(r"C:\game\gallery", exist_ok=True)
    
    # Create test stimuli
    with open(r"C:\game\gallery\poem.txt", "w", encoding="utf-8") as f:
        f.write("Love and hope are the light in the dark void.")
        
    with open(r"C:\game\gallery\storm.mp3", "w") as f:
        f.write("fake mp3 content")
        
    sensorium = Sensorium()
    
    print("   Running Perception Cycles...")
    for i in range(10):
        p = sensorium.perceive()
        if p:
            print(f"   👁️ Sense: {p['sense'].upper()} | Desc: {p['description']}")
            if p['sense'] == 'reading':
                print(f"      Sentiment: {p['sentiment']:.2f}")
            if p['sense'] == 'hearing':
                print(f"      Mood: {p['mood']}")
        else:
            print("   . (Nothing new)")
        time.sleep(0.5)

if __name__ == "__main__":
    test_sensorium()
