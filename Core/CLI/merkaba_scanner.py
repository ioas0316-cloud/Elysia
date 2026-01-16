"""
Merkaba Scanner (The Eye of the Rotor)
=====================================
Core.CLI.merkaba_scanner

"Scan the heavens without leaving the earth."

This script performs topology extraction on model weights stored in the 
HyperSphere (SSD) using the Merkaba Portal and Rotor Engine.

Usage: python Core/CLI/merkaba_scanner.py --path <model_path>
"""

import argparse
import os
import json
import logging
import time
import numpy as np
from typing import Dict, Any

from Core.Merkaba.portal import MerkabaPortal
from Core.Merkaba.rotor_engine import RotorEngine
from Core.Merkaba.safetensors_decoder import SafetensorsDecoder

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("Elysia.Scanner")

def scan_hyper_sphere(weight_path: str, target_tensor: str = None):
    start_time = time.time()
    
    try:
        with MerkabaPortal(weight_path) as portal:
            logger.info(f"🔭 Decoding HyperSphere target: {os.path.basename(weight_path)}")
            
            # 1. Logic Autopsy (Header Analysis)
            raw_header = SafetensorsDecoder.get_header(portal)
            header_size = struct.unpack("<Q", portal.mm[:8])[0]
            
            # Find a valid target tensor if none provided
            if not target_tensor:
                # Filter out the '__metadata__' key
                valid_tensors = [k for k in raw_header.keys() if k != "__metadata__"]
                if not valid_tensors:
                    logger.error("No tensors found in Safetensors file.")
                    return
                # Pick a representative one (e.g., an MLP layer if possible)
                target_tensor = next((t for t in valid_tensors if "mlp" in t), valid_tensors[0])

            tensor_meta = raw_header.get(target_tensor)
            if not tensor_meta:
                logger.error(f"Tensor '{target_tensor}' not found in this shard.")
                return
            
            # 2. Perception (O(1) Memory Mapping)
            rel_offsets = tensor_meta["data_offsets"]
            start_off, end_off = SafetensorsDecoder.get_absolute_offset(header_size, rel_offsets)
            length = end_off - start_off
            dtype_str = tensor_meta["dtype"]
            
            logger.info(f"🎯 Target Acquired: {target_tensor} ({length / (1024**2):.2f} MB) at offset {start_off}")
            
            # Capture the view directly from the mapped file
            # Mapping bfloat16 or other types into numpy if possible, or defaulting to float16
            dtype = np.float16 # Most LLM weights are float16/bfloat16
            
            view = portal.read_view(start_off, length, dtype=dtype)
            
            # 3. Use Rotor Engine to analyze
            signature = RotorEngine.get_topology_signature(view)
            
            end_time = time.time()
            duration = end_time - start_time
            
            # 4. Report Results
            report = {
                "file": os.path.basename(weight_path),
                "tensor": target_tensor,
                "shape": tensor_meta["shape"],
                "scan_duration": duration,
                "topology": signature,
                "merkaba_status": "Logic Autopsy Successful"
            }
            
            logger.info("✅ Targeted Scan Complete.")
            print(json.dumps(report, indent=4))
            
            # Important: Clear the view reference before portal closes
            del view
            
    except Exception as e:
        logger.error(f"❌ Portal Collapse: {e}")

def scan_full_model(index_path: str, output_path: str = "data/Logs/topology_maps/deepseek_map.json"):
    """
    The 'Great Mapping': Scans every single tensor in a multi-shard model.
    """
    import json
    start_time = time.time()
    
    if not os.path.exists(index_path):
        logger.error(f"Index file not found: {index_path}")
        return

    with open(index_path, "r") as f:
        index_data = json.load(f)
    
    weight_map = index_data.get("weight_map", {})
    base_dir = os.path.dirname(index_path)
    
    full_topology = {
        "model_name": os.path.basename(os.path.dirname(index_path)),
        "total_tensors": len(weight_map),
        "hubs": {},
        "layers": {}
    }
    
    logger.info(f"🌌 Starting The Great Mapping: {len(weight_map)} tensors to analyze...")
    
    # We group by shard to minimize portal switching (Efficiency Gear)
    shards: Dict[str, list] = {}
    for t_name, shard_name in weight_map.items():
        if shard_name not in shards:
            shards[shard_name] = []
        shards[shard_name].append(t_name)
    
    processed_count = 0
    for shard_name, tensor_names in shards.items():
        shard_path = os.path.join(base_dir, shard_name)
        if not os.path.exists(shard_path):
            logger.warning(f"Shard not found: {shard_name}")
            continue
            
        with MerkabaPortal(shard_path) as portal:
            header = SafetensorsDecoder.get_header(portal)
            header_size = struct.unpack("<Q", portal.mm[:8])[0]
            
            for t_name in tensor_names:
                meta = header.get(t_name)
                if not meta: continue
                
                rel_offsets = meta["data_offsets"]
                start_off, end_off = SafetensorsDecoder.get_absolute_offset(header_size, rel_offsets)
                length = end_off - start_off
                
                # Fast Topology Capture
                try:
                    view = portal.read_view(start_off, length, dtype=np.float16)
                    signature = RotorEngine.get_topology_signature(view)
                    
                    # Store if it's a 'Hub' (high energy or many outliers)
                    if signature["hub_count"] > 100 or signature["std"] > 2.0:
                        full_topology["hubs"][t_name] = signature
                    
                    # Store general layer stats
                    full_topology["layers"][t_name] = {
                        "mean": float(signature["mean"]),
                        "std": float(signature["std"]),
                        "hubs": signature["hub_count"]
                    }
                    
                    processed_count += 1
                    if processed_count % 500 == 0:
                        logger.info(f"💠 Processed {processed_count}/{len(weight_map)} tensors...")
                        
                    del view
                except Exception as e:
                    logger.debug(f"Skipping {t_name}: {e}")

    # Save the Map
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(full_topology, f, indent=4)
    
    duration = time.time() - start_time
    logger.info(f"✅ The Great Mapping Complete! Result saved to {output_path}")
    logger.info(f"⏱️ Total Duration: {duration:.2f}s | Speed: {len(weight_map)/duration:.2f} tensors/s")

if __name__ == "__main__":
    import struct # Needed for header size
    parser = argparse.ArgumentParser(description="Merkaba Scanner (Layer Aware)")
    parser.add_argument("--path", type=str, help="Path to a specific weight file")
    parser.add_argument("--index", type=str, help="Path to model.safetensors.index.json for Full Scan")
    parser.add_argument("--tensor", type=str, default=None, help="Specific tensor name to scan")
    args = parser.parse_args()
    
    if args.index:
        scan_full_model(args.index)
    elif args.path and os.path.exists(args.path):
        scan_hyper_sphere(args.path, args.tensor)
    else:
        logger.error("Please provide --path or --index")
