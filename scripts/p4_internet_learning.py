
"""
P4 Internet Learning Script
===========================
"Elysia reading the world."

This script triggers the P4SensorySystem to fetch emotional content
and "absorbs" it into Elysia's state.
"""

import sys
import json
import logging
from pathlib import Path
import random

# Setup paths
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.append(str(PROJECT_ROOT))
sys.path.append(str(PROJECT_ROOT / "Core" / "Sensory"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("P4_Learning")

try:
    from Core.Sensory.p4_sensory_system import P4SensorySystem
except ImportError as e:
    logger.error(f"Failed to import P4SensorySystem: {e}")
    sys.exit(1)

def run_learning_session(target_emotion: str):
    print(f"\n🧠 STARTING P4 LEARNING SESSION: {target_emotion}")
    print("=" * 60)
    
    p4 = P4SensorySystem()
    
    # 1. Search
    print(f"\n🔍 Searching for '{target_emotion}' content...")
    results = p4.fetch_emotional_content(target_emotion)
    
    if not results:
        print("❌ No results found.")
        return
        
    print(f"✅ Found {len(results)} potential sources.")
    for idx, res in enumerate(results):
        print(f"  [{idx+1}] {res['title']} ({res['url']})")
        
    # 2. Absorb (Simulate selection of first result)
    target_url = results[0]['url']
    print(f"\n📖 Absorbing text from: {target_url}")
    
    # NOTE: Since our "search" returns fake URLs for sites like PoetryFoundation,
    # and "fetch_content" tries to REAL fetch them, it might fail (404) because we constructed fake search URLs.
    # FOR DEMONSTRATION, we will use a real reachable URL if the search result looks fake, 
    # OR we rely on the robustness of P4 to handle 404s gracefully.
    # Let's try to actually fetch a real sample if possible.
    
    real_sample_url = "https://www.gutenberg.org/files/11/11-0.txt" # Alice in Wonderland (public domain)
    # We'll use this if the generated URL fails, but let's try the generated one first to see the system in action.
    
    absorption = p4.absorb_text(target_url)
    
    if absorption.get("status") == "failed":
        print(f"⚠️ Fetch failed ({absorption.get('reason')}). trying fallback real URL...")
        absorption = p4.absorb_text(real_sample_url)
    
    if absorption.get("status") == "success":
        text_preview = absorption['preview']
        print(f"\n✨ SUCCESSFULLY ABSORBED {absorption['length']} WORDS!")
        print(f"📝 Preview:\n{text_preview}\n")
        
        # 3. Update State (Simulated)
        print("💾 Updating Elysia's Emotional State...")
        state_path = PROJECT_ROOT / "elysia_state.json"
        if state_path.exists():
            with open(state_path, 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            # Update mood
            state["mood"] = f"Resonating with {target_emotion}"
            state["last_absorbed_url"] = absorption['url']
            
            with open(state_path, 'w', encoding='utf-8') as f:
                json.dump(state, f, indent=2)
            print("✅ State updated.")
            
    else:
        print("❌ Final extraction failed.")

if __name__ == "__main__":
    emotions = ["Joy", "Melancholy", "Hope", "Wonder"]
    target = sys.argv[1] if len(sys.argv) > 1 else random.choice(emotions)
    run_learning_session(target)
