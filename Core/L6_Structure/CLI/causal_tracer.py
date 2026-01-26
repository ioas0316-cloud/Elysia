"""

Causal Tracer (The Path Finder)

===============================

Core.L6_Structure.CLI.causal_tracer



"Follow the light of reason through the neural fog."



This tool traces the causal flow of a signal through the mapped hubs

of the DeepSeek model without full inference.

"""



import argparse

import os

import json

import numpy as np

import logging

from Core.L6_Structure.M1_Merkaba.simulator import RotorSimulator



logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

logger = logging.getLogger("Elysia.Tracer")



def trace_path(index_path: str, map_path: str, prompt: str):

    logger.info(f"✨ Tracing causal path for prompt: '{prompt}'")

    

    with open(map_path, 'r') as f:

        topology = json.load(f)

    

    hubs = topology.get("hubs", {})

    if not hubs:

        logger.error("No hubs found in map.")

        return



    simulator = RotorSimulator(index_path)

    

    # 1. Simple Tokenization (Placeholder for Phase 7 Tokenizer)

    # We create a pseudo-embedding for the prompt

    d_model = 2048

    initial_state = np.zeros(d_model)

    for char in prompt:

        np.random.seed(ord(char))

        initial_state += np.random.randn(d_model)

    initial_state /= np.linalg.norm(initial_state)

    

    current_state = initial_state

    

    # 2. Causal Propagation through Hubs

    # We trace the story of the prompt through the brain's depths

    target_tiers = ["embed", "layers.0.", "layers.5.", "layers.13.", "layers.26.", "lm_head"]

    

    logger.info("?  Path Selection:")

    

    for tier in target_tiers:

        # Find a hub in this tier (Case sensitive match)

        match = next((h for h in hubs.keys() if tier in h), None)

        

        if not match:

            # If no hub in top 200, try the general layer map

            match = next((h for h in topology.get("layers", {}).keys() if tier in h), None)

            

        if not match:

            print(f"WARN: Tier {tier} not found.")

            continue

        

        try:

            # Ignite the hub

            response = simulator.ignite_hub(match, current_state)

            

            energy = np.mean(np.abs(response))

            abs_resp = np.abs(response)

            sparsity = np.sum(abs_resp > np.mean(abs_resp)) / len(response)

            

            logic_type = "Logical Gate (Specific)" if sparsity < 0.4 else "Semantic Cloud (Broad)"

            

            print(f"PATH: {tier} -> {match} | Energy: {energy:.4f} | Essence: {logic_type}")

            

            # Update state

            new_state = response[:d_model] if len(response) >= d_model else response

            current_state = (current_state + new_state) / 2.0

            current_state /= (np.linalg.norm(current_state) + 1e-6)

            

        except Exception as e:

            print(f"FAIL: Tier {tier} -> {e}")



    print("--- TRACE COMPLETE ---")



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Causal Tracer")

    parser.add_argument("--index", type=str, required=True, help="Path to model index")

    parser.add_argument("--map", type=str, required=True, help="Path to topology map")

    parser.add_argument("--prompt", type=str, default="Elysia, what is the principle of life?", help="Input prompt")

    args = parser.parse_args()

    

    trace_path(args.index, args.map, args.prompt)
