"""
           (Quick Start)
================================

   AI +                

       :
1. Ollama      
2.             
3.            

  :
    python quick_start_local_ai.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from Core.S1_Body.L5_Mental.Reasoning_Core.Intelligence.ollama_bridge import ollama
from Core.S1_Body.L1_Foundation.Foundation.korean_wave_converter import korean_wave
from Core.S1_Body.L1_Foundation.Foundation.ether import ether, Wave

print("\n" + "="*70)
print("  Elysia    AI +          Quick Start")
print("="*70)

# ============================================================================
# 1. Ollama      
# ============================================================================

print("\n     1: Ollama      ")
print("-" * 70)

if not ollama.is_available():
    print("  Ollama            .")
    print("\n       :")
    print("   1.       'ollama serve'   ")
    print("   2.    Ollama     ")
    print("   3.        : 'ollama pull llama3.2:3b'")
    print("\n         : docs/LOCAL_LLM_SETUP_GUIDE.md")
    sys.exit(1)

print("  Ollama      !")

models = ollama.list_models()
if not models:
    print("            .")
    print("  'ollama pull llama3.2:3b'        .")
    sys.exit(1)

print(f"           : {', '.join(models)}")

# ============================================================================
# 2.             
# ============================================================================

print("\n     2:             ")
print("-" * 70)

#        
test_messages = [
    ("     ", "  "),
    ("   Elysia   ", "  "),
    ("       ", "  "),
]

print("          :")
for text, emotion in test_messages:
    wave = korean_wave.korean_to_wave(text, emotion=emotion)
    print(f"  '{text}' ({emotion})")
    print(f"         : {wave.frequency:.1f}Hz")
    print(f"        : {wave.phase}")

# ============================================================================
# 3.    AI       
# ============================================================================

print("\n     3:    AI       ")
print("-" * 70)

# Elysia         
elysia_system = """    Elysia   .
                  ,
                    .
                     ."""

#       
test_question = "  ?       ?         ."

print(f"  : {test_question}")
print("    ...")

response = ollama.chat(
    test_question,
    system=elysia_system,
    temperature=0.8
)

print(f"\nElysia: {response}")

# ============================================================================
# 4.        :           AI          
# ============================================================================

print("\n     4:        ")
print("-" * 70)

#        (  )
user_input = "      .      ?"
user_emotion = "  "

print(f"   : {user_input} ({user_emotion})")

# 1.           
user_wave = korean_wave.korean_to_wave(
    user_input,
    emotion=user_emotion,
    meaning="  "
)

print(f"         : {user_wave.frequency:.1f}Hz")

# 2.     Ether    
ether.emit(user_wave)
print(f"    Ether     ")

# 3. AI      
ai_response = ollama.chat(
    user_input,
    system=elysia_system,
    temperature=0.9  #          
)

print(f"\nElysia: {ai_response}")

# 4. AI           
response_wave = korean_wave.korean_to_wave(
    ai_response[:50],  #    50  
    emotion="  ",
    meaning="  "
)

print(f"         : {response_wave.frequency:.1f}Hz")

# 5.         
emotion_diff = abs(user_wave.frequency - response_wave.frequency)
print(f"          : {max(0, 100 - emotion_diff):.1f}%")

# ============================================================================
# 5.      
# ============================================================================

print("\n     5:      ")
print("-" * 70)

import time

#         
start = time.time()
quick_response = ollama.chat(
    "  ?",
    system="        .",
    max_tokens=50
)
elapsed = time.time() - start

print(f"     : {elapsed:.2f} ")
print(f"     : {len(quick_response)}   ")
print(f"  : {len(quick_response)/elapsed:.1f}   / ")

# ============================================================================
#   
# ============================================================================

print("\n" + "="*70)
print("           !")
print("="*70)

print("\n       !")
print("Elysia        AI       !")
print("                  !")

print("\n       :")
print("  1. living_elysia.py         AI   ")
print("  2.        -         ")
print("  3.              ")

print("\n     :")
print("  -    : docs/LOCAL_LLM_SETUP_GUIDE.md")
print("  - Ollama Bridge: Core/Intelligence/ollama_bridge.py")
print("  -      : Core/Language/korean_wave_converter.py")

print("\n  \"                  !\"")
print("="*70 + "\n")
