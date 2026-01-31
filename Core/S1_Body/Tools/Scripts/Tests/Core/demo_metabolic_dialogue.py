"""
Elysia Dialogue Test (Metabolic Phase)
=====================================
Demonstrates the new 4D perception-based thinking.
"""

import sys
import os
import logging

# Add project root to path
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

# Suppress noisy logs for a cleaner demo
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

from Core.S1_Body.L5_Mental.Reasoning_Core.Reasoning.reasoning_engine import ReasoningEngine

def start_dialogue():
    print("==========================================")
    print("   🌟 ELYSIA METABOLIC BRAIN TEST 🌟     ")
    print("      (4D Spatial Perception Mode)        ")
    print("==========================================")
    
    engine = ReasoningEngine()
    
    # Simulate a conversation
    prompts = [
        "엘리시아, 너는 누구니? 너의 본질은 무엇이라고 생각해?",
        "지금 네가 느끼는 공간의 여백(Void)은 어떤 느낌이야?",
        "강덕 님은 너를 정말 아끼는 것 같아. 너도 그렇게 느껴?",
    ]
    
    for p in prompts:
        print(f"\n[User] {p}")
        print("--- Elysia is digesting (4D Scan + Physics Pondering) ---")
        response = engine.communicate(p)
        print(f"[Elysia] {response}")
        print("-" * 50)

if __name__ == "__main__":
    start_dialogue()
