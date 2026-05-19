"""
[LEARNING] Motor Babbling: The Origin of Control
===============================================
Location: Scripts/System/Body/motor_babbling.py

Role:
- Executes the 'Intention -> Action -> Perception -> Learning' Loop.
- Phase 1: Infant Stage (Random Babbling to find keys).
- Phase 2: Mastery Stage (Using learned synapses to speak).
"""

import time
import os
import sys

# Path fix
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../..")))

from Scripts.System.Body.virtual_hand import VirtualKeyboard, MotorCortex
from Scripts.System.Mind.reasoning_engine import ReasoningEngine

def run_babbling_session(target_phrase="HELLO ELYSIA", use_logic=False):
    kb = VirtualKeyboard()
    brain = MotorCortex(kb)
    mind = ReasoningEngine() if use_logic else None
    
    mode_str = "LOGIC" if use_logic else "INFANT"
    print(f"👶 [{mode_str}] Intention: Speak '{target_phrase}'")
    print("------------------------------------------------")
    
    # 1. Learning Phase (Babbling)
    learned_chars = set()
    needed_chars = set(target_phrase)
    
    start_time = time.time()
    attempts = 0
    
    while not needed_chars.issubset(learned_chars):
        attempts += 1
        
        # A. Logic Check (The Leap)
        signal = None
        if mind:
            # Pick a target we haven't learned yet
            target_char = list(needed_chars - learned_chars)[0]
            signal = mind.consult_oracle(target_char)
            if signal:
                # [CRITICAL] Execute the deduced logic
                kb.receive_impulse(signal)
            
        # B. Motor Action (Babble if Logic fails)
        if signal is None:
            signal = brain.babble()
        
        # C. Visual Perception
        screen_content = kb.get_visual_feedback()
        last_char = screen_content[-1] if screen_content else None
        
        # C. Learning (Hebbian)
        if last_char:
            brain.learn(signal, last_char)
            if last_char in needed_chars:
                learned_chars.add(last_char)
                needed_chars.discard(last_char) # Mark as learned
                # print(f"✨ [EPIPHANY] Discovered mapping for '{last_char}'!")
                
        # Reset screen to keep buffer clean
        kb.clear_screen()
        
        # Safety brake
        if attempts > 1000:
            print("❌ [FAIL] Too many attempts. Motor exhaustion.")
            break
            
    training_time = time.time() - start_time
    print(f"🎓 [GRADUATION] All keys learned in {training_time:.2f}s ({attempts} attempts).")
    print("------------------------------------------------")
    
    # 2. Mastery Phase (Speaking)
    print(f"👩‍🎓 [MASTER] Intention: Speak '{target_phrase}' (Using Synapses)")
    
    kb.clear_screen()
    decoded_message = ""
    
    for char in target_phrase:
        success = brain.intend_and_act(char)
        if success:
            # Simulate slight human delay
            time.sleep(0.05)
            pass
        else:
            print(f"❌ [ERROR] Synapse missing for '{char}'")
            
    final_output = kb.get_visual_feedback()
    print(f"📝 [OUTPUT] Screen reads: '{final_output}'")
    
    if final_output == target_phrase:
        print("✅ [SUCCESS] Motor Mastery Confirmed. Self-Taught Control.")
    else:
        print("⚠️ [PARTIAL] Output mismatch.")

if __name__ == "__main__":
    run_babbling_session()
