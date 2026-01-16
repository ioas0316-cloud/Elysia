"""
Logic Autopsy (Forensic Engine)
===============================
Core.CLI.logic_autopsy

"Patterns lie. Principles endure."

This script analyzes the top 1% 'Golden Hubs' to extract their 
principled logic gates.
"""

import argparse
import os
import json
import logging
import numpy as np
from Core.Merkaba.simulator import RotorSimulator

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Elysia.Autopsy")

def perform_autopsy(index_path: str, map_path: str, target_tensor: str = None):
    logger.info("Starting Logic Autopsy...")
    
    with open(map_path, 'r') as f:
        topology = json.load(f)
    
    hubs = topology.get("hubs", {})
    layers = topology.get("layers", {})
    
    if not target_tensor:
        if not hubs:
            logger.error("No Golden Hubs found in map. Run scan first.")
            return
        target_tensor = max(hubs.items(), key=lambda x: x[1]['std'])[0]
    
    logger.info(f"Target: {target_tensor}")
    
    simulator = RotorSimulator(index_path)
    
    d_model = 2048 # Default for Lite version
    probe = np.random.randn(d_model)
    probe /= np.linalg.norm(probe)
    
    try:
        response = simulator.ignite_hub(target_tensor, probe)
        
        activation_mean = np.mean(np.abs(response))
        activation_sparsity = np.sum(np.abs(response) > activation_mean) / len(response)
        
        logger.info(f"Response Stats: Mean={activation_mean:.4f}, Sparsity={activation_sparsity:.2f}")
        
        if activation_sparsity < 0.3:
            principle = "High-Specificity Selector (Logic Gate)"
        elif activation_sparsity > 0.7:
            principle = "Uniform Harmonic (Global Bias)"
        else:
            principle = "Broad-Spectrum Integrator (Context Mixer)"
            
        logger.info(f"Extracted Principle: {principle}")
        
    except Exception as e:
        logger.error(f"❌ Autopsy Failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Logic Autopsy")
    parser.add_argument("--index", type=str, required=True, help="Path to model index")
    parser.add_argument("--map", type=str, required=True, help="Path to topology map")
    parser.add_argument("--tensor", type=str, default=None, help="Specific tensor name to analyze")
    args = parser.parse_args()
    
    perform_autopsy(args.index, args.map, args.tensor)
