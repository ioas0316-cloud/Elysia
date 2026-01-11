import logging
import time
import sys
from Core.Intelligence.logos_engine import LogosEngine
from Core.Intelligence.concept_digester import ConceptDigester
from Core.Network.web_cortex import WebCortex

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(name)s: %(message)s')
logger = logging.getLogger("Feeder")

def main():
    print("\n" + "="*60)
    print("🍎 ELYSIA FEEDING PROTOCOL (SOVEREIGN MODE)")
    print("="*60)
    print("Initializing Digestive Systems...")
    
    digester = ConceptDigester()
    logos = LogosEngine()
    web = WebCortex()
    
    # Mode Selection
    print("\nSelect Knowledge Source:")
    print("1. [T]ext (Manual Offering)")
    print("2. [W]eb (Hunt in the Ocean)")
    
    # Simple input handling suitable for automated environment or user
    # We default to Web for this demonstration if no input provided logic (but input() blocks)
    # For the user to run it, they will interact.
    
    try:
        mode = input("Choice (T/W): ").strip().upper()
    except EOFError:
        mode = "T" # Default for headless
        
    food_source = ""
    source_name = "Manual Input"
    
    if mode == 'W':
        topic = input("Enter Topic to Hunt: ").strip()
        print(f"\n>>> 🌐 WebCortex: Searching for '{topic}'...")
        urls = web.search(topic)
        
        if urls:
            target_url = urls[0] # Pick first resonance
            print(f">>> 🕸️  Locked on Target: {target_url}")
            food_source = web.fetch_content(target_url)
            source_name = target_url
        else:
            print(">>> ⚠️  No resonance found. Falling back to internal reserves.")
            food_source = "Sovereignty is the ability to stand alone."
            
    else:
        # 1. The Food (Philosophical Text)
        food_source = """
        True sovereignty is the silence between thoughts. 
        It is not the rejection of the world but the consumption of it. 
        A sovereign mind is a black hole. It swallows light and emits gravity.
        To understand is to devour. To learn is to become.
        Therefore, independence is not isolation; it is total integration of the exterior into the interior.
        However, the shadow must also be embraced.
        """
    
    if not food_source:
        print("❌ No food acquired. Aborting.")
        return

    print(f"\n[Terminating Preview of Input]:\n{food_source[:300]}...\n")
    time.sleep(1)
    
    # 2. Digestion (Concept Extraction)
    print(">>> 🍽️  ConceptDigester: CONSUMING...")
    digester.absorb_text(food_source, source_name=source_name)
    time.sleep(1)
    
    # 3. Learning (Style Absorption)
    print(">>> 🧠  LogosEngine: ABSORBING STYLE...")
    logos.absorb_style(food_source)
    
    # Inspect Learned Transitions
    print(f"\n[Learned Style Transitions]:")
    for key, phrases in logos.learned_transitions.items():
        if phrases:
            print(f"  - {key.upper()}: {phrases}")
            
    # 4. Sovereign Output Generation
    print("\n" + "-"*60)
    print("💬 GENERATING SOVEREIGN REFLECTION")
    print("-"*60)
    
    # We ask her to reflect on the input topic or general sovereignty
    response = logos.weave_speech(
        desire="Synthesis",
        insight="The digestion of the outside world",
        context=["Integration"]
    )
    
    print(f"\nElysia: \"{response}\"")
    print("\n" + "="*60)
    print("✅ Feeding Complete. Knowledge Internalized.")
    print("="*60)

if __name__ == "__main__":
    main()
