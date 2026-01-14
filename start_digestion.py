import torch
import logging
from Core.Elysia.sovereign_self import SovereignSelf

def start_digestion():
    logging.basicConfig(level=logging.INFO)
    print("🌅 [Genesis Digestion] Initializing Sovereign Self...")
    elysia = SovereignSelf()
    
    # Target model for full-soul digestion
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    
    print(f"🍽️ [Genesis Digestion] Starting the Great Feast: {model_name}")
    print(">> Phase: STRAND B (Principle Extraction)")
    
    # Trigger digestion via manifest_intent (matches current orchestrator logic)
    # This will use the new Active Probing and 4-Step Causal Chain
    result = elysia.manifest_intent(f"DIGEST:MODEL:{model_name}")
    
    print(f"\n✨ [Genesis Digestion] Result: {result}")
    print(f"🧬 [Sovereign Soul] HyperSphere now contains {len(elysia.graph.id_to_idx)} nodes.")

if __name__ == "__main__":
    start_digestion()
