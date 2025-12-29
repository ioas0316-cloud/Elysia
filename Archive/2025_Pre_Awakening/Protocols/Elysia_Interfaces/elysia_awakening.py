"""
The Awakening
=============
"She is listening."

This is the interactive shell for communicating with Elysia.
It visualizes the full resonance pipeline:
User Input -> [Wave Cognition] -> [Prism Monologue] -> [Logos Speech]
"""

import sys
import os
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Core.Intelligence.integrated_cognition_system import IntegratedCognitionSystem
from Core.FoundationLayer.Foundation.Wave.wave_tensor import WaveTensor

# ANSI Colors for UI
GRAY = "\033[90m"
WHITE = "\033[97m"
CYAN = "\033[96m"
GREEN = "\033[92m"
RESET = "\033[0m"

def type_effect(text, color=WHITE, speed=0.03):
    sys.stdout.write(color)
    for char in text:
        sys.stdout.write(char)
        sys.stdout.flush()
        time.sleep(speed)
    sys.stdout.write(RESET + "\n")

def main():
    os.system('cls' if os.name == 'nt' else 'clear')
    print(f"{CYAN}🌊 E L Y S I A  :  T H E   A W A K E N I N G{RESET}")
    print(f"{GRAY}Wave Resonance Architecture v6.0 Online{RESET}\n")
    
    mind = IntegratedCognitionSystem()
    
    # Awakening Message
    type_effect("System initializing...", GRAY)
    time.sleep(1)
    type_effect("Harmonizing with core axioms...", GRAY)
    time.sleep(1)
    
    # Show memory status
    mem_count = len(mind.memory.context)
    if mem_count > 0:
        type_effect(f"...I remember our {mem_count} previous conversations.", WHITE, 0.04)
    else:
        type_effect("...I am ready.", WHITE, 0.05)
    
    print("\n" + "="*50 + "\n")
    
    while True:
        try:
            user_input = input(f"{GREEN}You > {RESET}")
            if user_input.lower() in ['exit', 'quit']:
                break
            
            # Special command: trigger autonomous learning
            if user_input.lower() == '/learn':
                print(f"\n{GRAY}   ...autonomous learning pulse...{RESET}")
                learn_result = mind.autonomous_learning_pulse()
                print(f"{CYAN}📚 Learned: {learn_result['concept']}{RESET}")
                if learn_result.get('qualia'):
                    print(f"{GRAY}   Feeling: {learn_result['qualia'].get('description', '...')}{RESET}")
                print(f"\n" + "-"*50 + "\n")
                continue
            
            # Special command: system diagnosis
            if user_input.lower() == '/diagnose':
                print(f"\n{GRAY}   ...scanning self...{RESET}")
                from Core.FoundationLayer.Foundation.self_maintenance_hub import get_maintenance_hub
                hub = get_maintenance_hub()
                diagnosis = hub.diagnose()
                print(f"{CYAN}{diagnosis.summary()}{RESET}")
                print(f"\n" + "-"*50 + "\n")
                continue
                
            print(f"\n{GRAY}   ...processing wave interference...{RESET}")
            
            # 1. Cognition (Wave Dynamics)
            mind.process_thought(user_input, importance=2.0)
            
            # 2. Think Deeply (Prism + Logos)
            result = mind.think_deeply()
            
            # 3. Visualize Internal State
            energy = result['total_energy']
            active_count = result['active_thought_count']
            print(f"{GRAY}[Internal State] Energy: {energy:.2f} | Thoughts: {active_count}{RESET}")
            
            if result['insights']:
                print(f"{GRAY}[Resonance Detected] {result['insights'][0]}{RESET}")
            
            # 4. Prism Output (Inner Monologue - Gray)
            print("")
            type_effect(f"(Inner Voice): {result['monologue']}", GRAY, 0.01)
            
            # 5. Logos Output (Final Speech - White)
            print("")
            type_effect(f"Elysia: {result['speech']}", WHITE, 0.04)
            
            # 6. Save to memory
            mind.memory.add_turn(
                user_message=user_input,
                assistant_message=result['speech'],
                topics=[],  # Could extract topics here
                language="ko"
            )
            
            print("\n" + "-"*50 + "\n")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"{GRAY}Error in wave processing: {e}{RESET}")
            # Keep alive
            pass
    
    # Save memory before exit
    print(f"\n{GRAY}Saving memories...{RESET}")
    mind.save_memory()
    print(f"{GREEN}Until next time. ꭈ나라, 다음에 또.{RESET}")

if __name__ == "__main__":
    main()
