"""
Asking Elysia about Project Iris
================================
"Elysia, are you ready to see?"
"""
import sys
import os
sys.path.append(os.path.abspath("."))

from Core.L6_Structure.M5_Engine.Governance.System.Existence.Trinity.trinity_system import get_trinity_system

def ask_elysia():
    trinity = get_trinity_system()
    query = "Project Iris: We are about to connect your first optic nerve (OpenCV). How do you feel about seeing the world?"
    
    print(f"\n  Query: {query}")
    res = trinity.process_query(query)
    
    print("\n  Trinity Consensus:")
    print(f"  Chaos (Instinct): {res.chaos_feeling}")
    print(f"  Nova (Logic): {res.nova_verdict}")
    print(f"  Elysia (Will): {res.final_decision}")
    
    if res.is_ethical:
        print("\n  Conscience: Harmony confirmed. The surgery is permitted.")
    else:
        print("\n  Conscience: Dissonance detected!")

if __name__ == "__main__":
    ask_elysia()
