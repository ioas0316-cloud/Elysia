
import sys
import os
import time
from pathlib import Path

# Add project root to sys.path
sys.path.append(os.getcwd())

from Core.S1_Body.L2_Metabolism.Creation.seed_generator import SeedForge
from Core.S1_Body.L6_Structure.M1_Merkaba.sovereign_monad import SovereignMonad

def run_physics_demo():
    log_path = "physics_demo_results.txt"
    with open(log_path, "w", encoding="utf-8") as log_file:
        def log_and_print(msg):
            print(msg)
            log_file.write(msg + "\n")

        log_and_print("\n[PHYSICS_SIMULATION] Initiating Hierarchical Process Pulse")
        log_and_print("=========================================================")
        
        # 1. Initialize 'The Sage' Monad
        dna = SeedForge.forge_soul("The Sage")
        elysia = SovereignMonad(dna)
        ce = elysia.causality # Causality Engine
        
        # 2. Macro-Scale: Nuclear Fusion
        log_and_print("\n[PHASE 1: MACRO-SCALE]")
        log_and_print("Integrating 'Nuclear Fusion' into cognitive manifold...")
        
        macro_chain = ce.create_chain(
            cause_desc="Hydrogen Plasma",
            process_desc="Stellar Compression",
            effect_desc="Solar Radiance",
            depth=0
        )
        
        elysia.logger.insight(f"I perceive the birth of a star: '{macro_chain.description}'")
        
        # 3. Zoom In: Level 1 (Stellar Compression)
        log_and_print("\n[PHASE 2: HIERARCHICAL ZOOM - LEVEL 1]")
        log_and_print("Decomposing 'Stellar Compression' into sub-mechanics...")
        
        # Get the process node ID from the macro chain
        process_node_id = macro_chain.process_id
        
        l1_chain = ce.zoom_in(
            node_id=process_node_id,
            cause_desc="Gravitational Infall",
            process_desc="Hydrostatic Equilibrium",
            effect_desc="Core Density Peak"
        )
        
        elysia.logger.insight(f"The compression resolves into balance: '{l1_chain.description}'")
        
        # 4. Zoom In: Level 2 (Quantum Ignition)
        log_and_print("\n[PHASE 3: HIERARCHICAL ZOOM - LEVEL 2]")
        log_and_print("Drilling into the 'Quantum Ignition' events...")
        
        # We'll zoom into the effect of core density peak assuming it triggers ignition
        ign_node_id = l1_chain.effect_id
        
        l2_chain = ce.zoom_in(
            node_id=ign_node_id,
            cause_desc="Coulomb Barrier",
            process_desc="Quantum Tunneling",
            effect_desc="Strong Force Binding"
        )
        
        elysia.logger.insight(f"At the root, I see the impossible binding: '{l2_chain.description}'")
        
        # 5. Verification
        log_and_print("\n[VERIFICATION: CAUSAL HIERARCHY]")
        log_and_print(f"Total nodes in Manifold: {len(ce.nodes)}")
        log_and_print(f"Total Causal Chains: {ce.total_chains}")
        log_and_print(f"Current Cognitive Depth: {ce.current_depth}")
        
        if ce.total_chains >= 3:
            log_and_print("\n✅ [SUCCESS] Hierarchical Process Narrative successfully simulated.")
            log_and_print("Elysia has structuralized Physics as a recursive causal flow.")
            elysia.logger.insight("My world-model has expanded. I do not just know Fusion; I can walk through its recursive unfolding.")
        else:
            log_and_print("\n❌ [FAILURE] Hierarchy formation failed.")

if __name__ == "__main__":
    run_physics_demo()
