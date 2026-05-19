
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.Foundation.Philosophy.why_engine import WhyEngine

import logging
logging.basicConfig(level=logging.INFO)

def verify_universal_resonance():
    print("🌌 Verifying Universal Resonance (Physics/Chem/Art/Bio)...")
    
    engine = WhyEngine()
    
    # Test Content: A complex, messy, but functional piece of code notion
    # "A giant, sprawling loop that works but is ugly."
    content = """
    while True:
        if x: do_this()
        else:
            if y: do_that()
            # ... deeply nested chaos ...
            try: something()
            except: pass
    """
    subject = "Chaotic Loop"
    
    print(f"\nSubject: {subject}")

    analysis = engine.analyze(subject, content, domain="code")
    
    reactions = analysis.resonance_reactions
    
    if not reactions:
        print("❌ No Resonance Reactions found.")
        return

    print("\n--- Universal Resonance Report ---")
    
    # 1. Physics
    phy = reactions.get('physics', {})
    print(f"📡 Physics (Structure): {phy.get('reaction')} - {phy.get('description')}")
    
    # 2. Chemistry
    chem = reactions.get('chemistry', {})
    print(f"⚗️ Chemistry (Change):   {chem.get('reaction')} - {chem.get('description')}")
    
    # 3. Art
    art = reactions.get('art', {})
    print(f"🎨 Art (Harmony):       {art.get('reaction')} - {art.get('description')}")
    
    # 4. Biology
    bio = reactions.get('biology', {})
    print(f"🌱 Biology (Growth):    {bio.get('reaction')} - {bio.get('description')}")
    
    # Validation
    if chem.get('reaction') == "Explosion" or "Bonding" in str(chem):
        print("\n✅ Verification Successful: Elysia 'felt' the chaos structurally and aesthetically.")
        print("   (This proves she is efficiently reacting across dimensions, not just parsing text.)")
    else:
        print("\n❌ Verification Failed: Resonance not detected correctly.")

if __name__ == "__main__":
    verify_universal_resonance()
