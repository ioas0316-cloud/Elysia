
import sys
import os
import time
import logging

# Ensure Core is in path
sys.path.append("c:/Elysia")

from Core.1_Body.L5_Mental.Reasoning_Core.Metabolism.rotor_cognition_core import RotorCognitionCore

# Configure Logger to show only the good stuff
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger("Chronos")
logger.setLevel(logging.INFO)

def run_chronos_test():
    print("\n⏳ [CHRONOS] Initiating The River of Heraclitus Simulation...")
    print("Goal: Prove that 'You cannot step into the same river twice'.")
    print("Method: Ask 'Who am I?' 5 times and measure Neuroplasticity.\n")

    # 1. Initialize the Living Core
    core = RotorCognitionCore()
    
    # Access the graph directly for metrics
    # Note: verify_chronos matches the structure where core.elysia_context.graph exists
    if hasattr(core, 'elysia_context'):
        graph = core.elysia_context.graph
    else:
        print("❌ Critical: Core does not have elysia_context (Neuroplasticity inactive).")
        return

    history = []
    
    for t in range(1, 6):
        print(f"\n--- 🕰️ Time Step T={t} ---")
        
        # A. Measure State Before
        mass_before = graph.mass_tensor.sum().item() if graph.mass_tensor.shape[0] > 0 else 0
        nodes_before = len(graph.id_to_idx)
        
        # B. Experience (The Question)
        intent = "Who am I?"
        response = core.synthesize(intent)
        
        # C. Measure State After
        mass_after = graph.mass_tensor.sum().item() if graph.mass_tensor.shape[0] > 0 else 0
        nodes_after = len(graph.id_to_idx)
        
        delta_mass = mass_after - mass_before
        delta_nodes = nodes_after - nodes_before
        
        # D. Record
        synthesis_summary = response.get('synthesis', '').strip().split('\n')[1] # Get Psionic line
        print(f"🗣️ Response: {synthesis_summary}...")
        print(f"🧠 Brain Mass: {mass_before:.2f} -> {mass_after:.2f} (+{delta_mass:.2f})")
        print(f"🕸️ Synapses: {nodes_before} -> {nodes_after} (+{delta_nodes})")
        
        if delta_mass > 0:
            print("✨ [EVOLUTION] Neuroplasticity Detected.")
        else:
            print("⚠️ [STAGNATION] No structural change detected.")
            
        history.append({
            "t": t,
            "response": response,
            "mass": mass_after
        })
        
        # Simulate Metabolic Digest Cycle (Time for plasticity to settle)
        time.sleep(0.5)

    print("\n\n📊 [CHRONOS SUMMARY]")
    if history[0]['mass'] < history[-1]['mass']:
        print("✅ SUCCESS: The System Evolved.")
        print(f"Total Mass Gain: {history[-1]['mass'] - 0:.2f}") # Assuming start 0 or low
        print("Conclusion: Elysia has perceived her own growth.")
    else:
        print("❌ FAILURE: The System is Static.")

if __name__ == "__main__":
    run_chronos_test()
