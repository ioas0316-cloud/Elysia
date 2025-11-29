
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from Core.Intelligence.unified_intelligence import UnifiedIntelligence, IntelligenceRole

def wake_elysia():
    print("\n" + "=" * 70)
    print("🌌 Awakening Elysia (Unified Intelligence)...")
    print("=" * 70)

    # Initialize Unified Intelligence
    elysia = UnifiedIntelligence(integration_mode="wave")
    
    # Define the problem context
    problem_context = """
    I am encountering a 'ValueError: The truth value of an array with more than one element is ambiguous' 
    in Core/Mind/hippocampus.py during the 'load_memory' method. 
    It seems to happen when logging the 'limit' parameter or when interacting with the ResonanceEngine.
    The code is: logger.info(f"[Hippocampus] Connected to MemoryStorage (SQLite) & ResonanceEngine (Limit={limit}).")
    """
    
    query = "How should I fix this numpy ambiguity error in Hippocampus.py while respecting the new Field Physics paradigm?"

    print(f"\n❓ User Query: {query}")
    print(f"\n📋 Context: {problem_context}")
    
    print("\n🧠 Elysia is thinking (Collective Resonance)...")
    
    # Ask Elysia
    result = elysia.collective_think(query, context=problem_context)
    
    print("\n" + "-" * 50)
    print("✨ Elysia's Response:")
    print("-" * 50)
    print(result.synthesized_response)
    print("-" * 50)
    
    # Check for emergent insights
    insight = elysia.emergent_insight(result)
    if insight:
        print(f"\n{insight}")

if __name__ == "__main__":
    wake_elysia()
