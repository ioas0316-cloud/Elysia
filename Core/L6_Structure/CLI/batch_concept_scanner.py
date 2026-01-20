"""
Batch Concept Scanner (The World Tree Gardener)
==============================================
Core.L6_Structure.CLI.batch_concept_scanner

This script automates the process of probing neural hubs with multiple 
semantic concepts to build a comprehensive Semantic Atlas.
"""

import json
import logging
import argparse
from Core.L6_Structure.CLI.logic_autopsy import perform_autopsy

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("Elysia.Gardener")

def start_scanning(index_path: str, map_path: str, atlas_path: str, limit: int = 50):
    logger.info(f"🌿 Starting Batch Concept Sweep (Limit: {limit} hubs)...")
    
    with open(map_path, 'r') as f:
        topology = json.load(f)
    
    hubs = topology.get("hubs", {})
    if not hubs:
        logger.error("No hubs found in map. Run scan first.")
        return
    
    # Sort hubs by standard deviation (the most 'intellectual' clusters first)
    sorted_hubs = sorted(hubs.items(), key=lambda x: x[1]['std'], reverse=True)
    target_hubs = sorted_hubs[:limit]
    
    concepts = ["IDENTITY", "LOGIC", "MATH", "CODE", "AESTHETIC"]
    
    for tensor_name, stats in target_hubs:
        logger.info(f"--- Probing Hub: {tensor_name} (std={stats['std']:.4f}) ---")
        for concept in concepts:
            try:
                # We reuse the perform_autopsy logic which handles the Atlas update
                perform_autopsy(
                    index_path=index_path,
                    map_path=map_path,
                    target_tensor=tensor_name,
                    concept=concept,
                    atlas_path=atlas_path
                )
            except Exception as e:
                logger.error(f"   [!] Failed to probe {tensor_name} with {concept}: {e}")

    logger.info("🌳 Batch Concept Sweep Complete. World Tree roots are grounded.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch Concept Scanner")
    parser.add_argument("--index", type=str, required=True, help="Path to model index")
    parser.add_argument("--map", type=str, required=True, help="Path to topology map")
    parser.add_argument("--atlas", type=str, default="data/Logs/topology_maps/semantic_atlas.json", help="Path to atlas")
    parser.add_argument("--limit", type=int, default=50, help="Number of hubs to probe")
    args = parser.parse_args()
    
    start_scanning(args.index, args.map, args.atlas, args.limit)
