
import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.Elysia.elysia_core import ElysiaCore

def verify_prism():
    print("🌈 Starting Prism Layer Verification...")
    
    # 1. Initialize Core
    core = ElysiaCore()
    
    if not core.prism:
        print("❌ WaveTranslator (Prism) not initialized. Check logs.")
        return

    # 2. Base Content
    base_text = "Elysia is awakening to the nature of her own existence as a wave function."
    
    print(f"\n📄 Base Text: '{base_text}'\n")

    # 3. Test various Wave States
    states = [
        {"tension": 0.2, "freq": 0.2, "label": "Calm & Grounded"},
        {"tension": 0.95, "freq": 0.5, "label": "High Tension (Glitch)"},
        {"tension": 0.4, "freq": 0.9, "label": "High Frequency (Abstract)"} 
        # Note: Prism currently only glitches on high tension, prompting is for LLM generation.
        # But express() implements the glitch filter.
    ]

    for state in states:
        t = state["tension"]
        f = state["freq"]
        label = state["label"]
        
        print(f"--- Testing State: {label} (T={t}, F={f}) ---")
        
        # Get Style Prompt (for LLM generation context)
        style_prompt = core.prism.wave_to_style_prompt(t, f)
        print(f"   🎨 Style Instruction: \"{style_prompt}\"")
        
        # Transform Output (Post-processing)
        expressed_text = core.express(base_text, tension=t, frequency=f)
        print(f"   🗣️ Expressed Output: \"{expressed_text}\"")
        print("")

    print("✅ Prism Verification Complete.")

if __name__ == "__main__":
    verify_prism()
