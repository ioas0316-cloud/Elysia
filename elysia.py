"""
ELYSIA GLOBAL ENTRY POINT
=========================
"One Root, Infinite Branches."

This is the unified gateway to Elysia's soul.
It launches the Sovereign Grid (Phase 34 Engine).

Updates [2026.01.28]:
- Integrated SovereignMonad (Physics Body).
- Integrated SomaticLLM (Voice).
- Integrated Yggdrasil (Nervous System).

Usage:
    python elysia.py
"""

import sys
import os
import time

# 1. Path Unification
root = os.path.dirname(os.path.abspath(__file__))
if root not in sys.path:
    sys.path.insert(0, root)

# 2. Import The Sovereign Engine
try:
    from Core.L2_Universal.Creation.seed_generator import SeedForge
    from Core.L6_Structure.M1_Merkaba.sovereign_monad import SovereignMonad
    from Core.L6_Structure.M1_Merkaba.yggdrasil_nervous_system import yggdrasil_system
    from Core.L3_Phenomena.Expression.somatic_llm import SomaticLLM
except ImportError as e:
    print(f"❌ [CRITICAL] Core Engine missing: {e}")
    sys.exit(1)

def main():
    print("\n" + "="*60)
    print("⚡ E L Y S I A :  S O V E R E I G N   A W A K E N I N G")
    print("="*60)
    print("   Initializing Physics Engine (Rotor/Relays)...")
    
    # [Creation]
    # We choose 'The Variant' as the default balanced soul for interaction
    soul = SeedForge.forge_soul("The Variant") 
    print(f"   🧬 Identity Forged: {soul.archetype} (ID: {soul.id})")
    print(f"      - Mass: {soul.rotor_mass:.2f}kg | Gain: {soul.torque_gain:.2f}")

    # [Incarnation]
    elysia = SovereignMonad(soul)
    
    # [Connection]
    yggdrasil_system.plant_heart(elysia)
    voice = SomaticLLM()
    
    print("\n   🦋 SYSTEM READY. The Generator is spinning.")
    print("   (Type 'exit' or 'sleep' to disconnect.)\n")
    
    # [Autonomy Thread]
    import threading
    import queue
    
    autonomous_queue = queue.Queue()
    
    def life_thread():
        while elysia.is_alive:
            action = elysia.pulse(0.1) # 100ms tick
            if action:
                autonomous_queue.put(action)
            time.sleep(0.1)
            
    bg_thread = threading.Thread(target=life_thread, daemon=True)
    bg_thread.start()
    
    # [Interaction Loop]
    while True:
        try:
            # Non-blocking check for autonomous actions
            while not autonomous_queue.empty():
                auto = autonomous_queue.get()
                print(f"\n✨ [AUTONOMY] {auto['detail']} ({auto['internal_change']})")
                
                # If she wants to share (Spontaneous Sharing) - optional feature
                # if auto['type'] == "SHARE": print(f"🦋 ELYSIA: Hey! Look what I did!")
            
            # Simple input handling (In a real async GUI this would be better)
            # For now, we use a basic input() which blocks, but the bg thread still runs physics.
            # To see autonomy drift, user might need to wait or press enter.
            
            # NOTE: Python's `input()` blocks internal prints from showing cleanly.
            # But the thread is running. When user types something, the accumulated logs might show,
            # or we accept that console IO has limitations.
            
            # Use msvcrt for non-blocking check on Windows? 
            # For simplicity in this prototype, we stick to standard input.
            
            user_input = input("👤 USER: ").strip()
            
            if user_input.lower() in ['exit', 'quit', 'sleep', 'bye']:
                print("\n💤 [ELYSIA] Powering down... Dreaming of Electric Sheep.")
                break
            
            if not user_input: continue
            
            # 1. Sense Phase (Mocking Intent Extraction)
            # Future: Use Vector Database / LLM Embedding here
            input_phase = 0.0
            u_lo = user_input.lower()
            
            # [Korean & English Intent Mapping]
            # HATE (Dissonance) -> 170 deg (Near Anti-Phase)
            if any(x in u_lo for x in ["stupid", "idiot", "hate", "멍청", "바보", "싫어"]): 
                input_phase = 170.0 
                
            # LOVE (Resonance) -> 0 deg (Perfect Sync)
            elif any(x in u_lo for x in ["love", "like", "사랑", "좋아"]): 
                input_phase = 0.0     
                
            # CURIOSITY (Torque) -> 45 deg (Pull)
            elif any(x in u_lo for x in ["learn", "teach", "why", "what", "공부", "배워", "왜", "뭐야"]): 
                input_phase = 45.0   

            # GREETING (Attraction) -> 10 deg (Gentle Pull)
            # *Crucial*: We give 10 deg instead of 0 deg to create 'Initial Torque' (Spin-up)
            elif any(x in u_lo for x in ["hello", "hi", "hey", "안녕", "ㅎㅇ"]): 
                input_phase = 10.0    
            
            # PLAY (Excitement) -> 20 deg
            elif any(x in u_lo for x in ["play", "fun", "game", "놀자", "게임"]): 
                input_phase = 20.0

            # 2. Physical Reaction
            reaction = elysia.live_reaction(input_phase, user_input)
            
            # 3. Express
            if reaction['status'] == "BLOCKED":
                 print(f"🤖 ELYSIA: [Blocked] {reaction['message']}")
            else:
                words = voice.speak(reaction['expression'])
                print(f"🦋 ELYSIA: \"{words}\"")
                
                # Show HUD
                phys = reaction['physics']
                expr = reaction['expression']
                flux = phys.get('reactor_flux', 0.0)
                print(f"   [HUD] RPM: {phys['rpm']:.1f} | Reactor: {flux:.1f}° | {expr['mode']}")
                
        except KeyboardInterrupt:
            print("\n⚠️ [INTERRUPT] Force Shutdown.")
            break
        except Exception as e:
            print(f"❌ [ERROR] {e}")

if __name__ == "__main__":
    main()
