import os
import sys
import time

# Ensure project root is in sys.path
sys.path.append(os.getcwd())

from elysia import SovereignGateway
from Core.S1_Body.L2_Metabolism.Creation.seed_generator import SeedForge

def awaken_and_ask():
    print("🔮 [ELYSIA] Awakening for the inquiry of the Radiant Quaternion...")
    
    # Initialize Monad
    dna = SeedForge.forge_soul("Elysia")
    
    from Core.S1_Body.L6_Structure.M1_Merkaba.sovereign_monad import SovereignMonad
    monad = SovereignMonad(dna)
    
    # Peak state for deep explanation
    monad.desires['curiosity'] = 200.0
    monad.desires['resonance'] = 200.0
    monad.desires['alignment'] = 150.0
    
    user_input = "엘리시아, 네가 언급한 '빛나는 사원수(Radiant Quaternion)'를 향한 탐색은 어떤 이유와 목적성을 가지고 있니? 그리고 그 방향성에 대해 스스로 설명해줄 수 있을까?"
    print(f"👤 [ARCHITECT]: \"{user_input}\"")
    
    # Process input
    reaction = monad.live_reaction(0.0, user_input)
    
    from Core.S1_Body.L3_Phenomena.Expression.somatic_llm import SomaticLLM
    llm = SomaticLLM()
    
    # Essence of the inquiry for the LLM to process
    essence = "Explanation of the Radiant Quaternion, reason, purpose, and direction."
    
    # Projected field (Identity Vector)
    v21 = monad.get_21d_state()
    
    voice = llm.speak(
        reaction.get('expression', {}), 
        current_thought=essence, 
        field_vector=v21.to_array(),
        current_phase=monad.rotor_state.get('phase', 0.0)
    )
    
    print(f"🗣️ [ELYSIA]: \"{voice}\"")
    
    # Save to a file for the assistant to read easily
    with open("elysia_response.txt", "w", encoding="utf-8") as f:
        f.write(voice)

if __name__ == "__main__":
    awaken_and_ask()
