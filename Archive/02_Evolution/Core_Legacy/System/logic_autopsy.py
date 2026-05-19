"""
Logic Autopsy (Forensic Engine)
===============================
Core.System.logic_autopsy

"Patterns lie. Principles endure."

This script analyzes the top 1% 'Golden Hubs' to extract their 
principled logic gates.
"""

import argparse
import os
import json
import logging
import numpy as np
import struct
from Core.Monad.simulator import RotorSimulator
from Core.Monad.semantic_atlas import SemanticAtlas
from Core.Monad.portal import MerkabaPortal
from Core.Monad.safetensors_decoder import SafetensorsDecoder

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Elysia.Autopsy")

def get_concept_probe(concept_name: str, dimension: int = 2048) -> np.ndarray:
    """
    Generates a deterministic semantic probe based on a concept name.
    """
    # Use the sum of characters as a seed for determinism
    seed = sum([ord(c) for c in concept_name])
    rng = np.random.default_rng(seed)
    probe = rng.standard_normal(dimension)
    probe /= np.linalg.norm(probe)
    return probe

def perform_autopsy(index_path: str, map_path: str, target_tensor: str = None, concept: str = None, atlas_path: str = None):
    logger.info("Starting Semantic Logic Autopsy...")
    
    with open(map_path, 'r') as f:
        topology = json.load(f)
    
    hubs = topology.get("hubs", {})
    
    if not target_tensor:
        if not hubs:
            logger.error("No Golden Hubs found in map.")
            return
        target_tensor = max(hubs.items(), key=lambda x: x[1]['std'])[0]
    
    logger.info(f"Targeting Neural Hub: {target_tensor}")
    
    simulator = RotorSimulator(index_path)
    atlas = SemanticAtlas(atlas_path) if atlas_path else None
    
    # Generate Probe (Detecting target dimension)
    try:
        # We need to know the shape to create the right probe
        shard_name = simulator.weight_map.get(target_tensor)
        if not shard_name:
            logger.error(f"Tensor '{target_tensor}' not found in weight map. available: {list(simulator.weight_map.keys())[:5]}...")
            return

        shard_path = os.path.join(simulator.base_dir, shard_name)
        with MerkabaPortal(shard_path) as portal:
            header = SafetensorsDecoder.get_header(portal)
            shape = header[target_tensor]["shape"]
            # For Linear(in, out), weight is (out, in). So we need 'in' dim.
            target_dim = shape[1] 
            
        probe_concept = concept if concept else "PURE_LOGIC"
        probe = get_concept_probe(probe_concept, dimension=target_dim)
        
        response = simulator.ignite_hub(target_tensor, probe)
        
        activation_mean = np.mean(np.abs(response))
        abs_resp = np.abs(response)
        sparsity = np.sum(abs_resp > np.mean(abs_resp)) / len(response)
        
        logger.info(f"Probing with Concept: {probe_concept}")
        logger.info(f"Response Stats: Mean={activation_mean:.4f}, Sparsity={sparsity:.2f}")
        
        # Classification Logic
        if sparsity < 0.3:
            principle = "Logical Gate (Specific Response)"
        elif sparsity > 0.7:
            principle = "Global Harmonic (Bias/Context)"
        else:
            principle = "Semantic Integrator (Concept Mixer)"
            
        logger.info(f"Extracted Principle: {principle}")

        if atlas and activation_mean > 0.1: # Threshold for tagging
            strength = float(activation_mean)
            atlas.tag_tensor(target_tensor, [probe_concept], strength=strength)
            atlas.save()
            logger.info(f"Atlas Updated: {target_tensor} marked with {probe_concept}")
            
    except Exception as e:
        logger.error(f"Autopsy Failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Logic Autopsy")
    parser.add_argument("--index", type=str, required=True, help="Path to model index")
    parser.add_argument("--map", type=str, required=True, help="Path to topology map")
    parser.add_argument("--tensor", type=str, default=None, help="Specific tensor name")
    parser.add_argument("--concept", type=str, default=None, help="Concept to probe with")
    parser.add_argument("--atlas", type=str, default="data/L6_Structure/Logs/topology_maps/semantic_atlas.json", help="Path to semantic atlas")
    args = parser.parse_args()
    
    perform_autopsy(args.index, args.map, args.tensor, args.concept, args.atlas)
