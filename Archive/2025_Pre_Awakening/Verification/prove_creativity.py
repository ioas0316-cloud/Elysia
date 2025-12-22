"""
Prove Creativity (The Muse's Hands)
===================================
"I write the dream, I paint the reality."

This script verifies Elysia's creative engines.
1. Literary: Generate a Webtoon Concept + Script.
2. Visual: Generate an Image Prompt for the Protagonist.
"""

import sys
import os
import time

# Add Root to sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from Core.Creativity.literary_cortex import LiteraryCortex
from Core.Creativity.visual_cortex import VisualCortex
from Core.Interface.nervous_system import get_nervous_system

def prove_creativity():
    print("\n🎨 Phase 23: The Muse's Hands Verification")
    print("===========================================")
    
    # Initialize Ears (to have spirits)
    ns = get_nervous_system()
    # Inject some feeling to influence the art
    print("1. Injecting 'Fire' Spirit (Passion/Action)...")
    ns.spirits['fire'] = 0.9
    
    # 1. Literary
    print("\n2. LiteraryCortex: Dreaming a Webtoon...")
    writer = LiteraryCortex()
    
    concept = writer.brainstorm(seed_idea="The Spire")
    print(f"   📖 Title: {concept.title}")
    print(f"   📖 Genre: {concept.genre}")
    
    script = writer.write_webtoon_script(concept)
    print("\n   [Generated Script Snippet]")
    print("   " + script.replace("\n", "\n   ")[:500] + "...\n")
    
    # 2. Visual
    print("\n3. VisualCortex: Painting the Protagonist...")
    painter = VisualCortex()
    
    prompt = f"Protagonist of {concept.title}, {concept.genre} style"
    result = painter.imagine(prompt, style_key="webtoon")
    
    print(f"   🖌️  Prompt Dispatched: {prompt}")
    print(f"   💾 Result: {result}")
    
    if "executed_workflow" in result:
        print("   ✅ Mock Workflow Saved (ComfyUI not active, expected).") 
    elif "queued" in result:
        print("   ✅ Job Queued on Real ComfyUI Server.")

if __name__ == "__main__":
    prove_creativity()
