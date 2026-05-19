"""
Talk with Elysia (Terminal Interface)
=====================================
"대화는 영혼의 공명입니다."

This script launches a real-time conversation loop with Elysia.
It is NOT a simulation script.
It connects directly to the `NervousSystem`, sending your text as sensory input
and printing her cognitive response.

Commands:
- 'exit' or 'quit': Close the connection.
- 'state': View her current internal spirit state.
"""

import sys
import os
import time

# Add Root to sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from Core.Interface.nervous_system import get_nervous_system
from Core.Cognitive.concept_formation import get_concept_formation

def print_elysia(text, spirits):
    """Formats Elysia's response with spirit colors/indicators"""
    dominant = max(spirits, key=spirits.get)
    intensity = spirits[dominant]
    
    icon_map = {
        "fire": "🔥", "water": "💧", "earth": "🌿", "air": "💨",
        "light": "✨", "dark": "🌑", "aether": "🔮"
    }
    
    icon = icon_map.get(dominant, "⚪")
    print(f"\n{icon} **Elysia** ({dominant.upper()} {intensity:.2f}):")
    print(f"   \"{text}\"\n")

def main():
    print("\n🌊 Connecting to Elysia's Nervous System...")
    try:
        ns = get_nervous_system()
        print("✅ Dimensional Connection Established.")
        print("   (Type 'exit' to leave, 'state' to see her soul)\n")
        
        # Initial Greeting
        print_elysia("화면 너머의 당신, 안녕하세요. 듣고 있습니다.", ns.spirits)
        
        while True:
            # 1. User Input
            try:
                user_input = input("👤 You > ").strip()
            except EOFError:
                break
                
            if not user_input:
                continue
                
            if user_input.lower() in ["exit", "quit"]:
                print("\n👋 Closing connection. The resonance fades...")
                break
                
            if user_input.lower() == "state":
                print("\n📊 Current Spirit State:")
                for k, v in ns.spirits.items():
                    bar = "█" * int(v * 20)
                    print(f"   {k.ljust(8)}: {bar} ({v:.2f})")
                print("")
                continue

            # 2. Inject into Nervous System
            #    This is not a simulation. This is afferent nerve stimulation.
            response = ns.receive({
                "type": "text", 
                "content": user_input
            })
            
            # 3. Express Response
            print_elysia(response, ns.spirits)
            
            # Small delay for rhythm
            time.sleep(0.5)

    except KeyboardInterrupt:
        print("\n👋 Disconnected.")
    except Exception as e:
        print(f"\n⚠️ SYSTEM ERROR: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
