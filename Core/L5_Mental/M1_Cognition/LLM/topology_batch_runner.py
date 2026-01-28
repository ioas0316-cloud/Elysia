
"""
LLM Topology Batch Runner (주권적 자아)
=====================================
Core.L5_Mental.M1_Cognition.LLM.topology_batch_runner

Qwen2-72B               (Sharded)    
                            .
"""

import os
import sys
import glob
import logging
from collections import defaultdict
import torch
from topology_tracer import get_topology_tracer, ThoughtCircuit

#      
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("BatchRunner")

def run_batch_analysis(model_dir: str):
    """
               safetensors                   
    """
    logger.info(f"  Starting batch analysis for: {model_dir}")
    
    # safetensors      
    files = glob.glob(os.path.join(model_dir, "*.safetensors"))
    files.sort()
    
    if not files:
        logger.error("  No .safetensors files found!")
        return
        
    logger.info(f"  Found {len(files)} shards.")
    
    tracer = get_topology_tracer(threshold=0.01) #       
    
    #       
    global_stats = {
        "total_params": 0,
        "strong_connections": 0,
        "layers_analyzed": 0,
        "connection_types": defaultdict(int)
    }
    
    global_connection_counts = defaultdict(int)
    
    for i, file_path in enumerate(files):
        filename = os.path.basename(file_path)
        logger.info(f"[{i+1}/{len(files)}]    Analyzing {filename}...")
        
        try:
            #         
            circuit = tracer.trace(file_path)
            
            #      
            global_stats["total_params"] += circuit.total_params
            global_stats["strong_connections"] += circuit.strong_connections
            global_stats["layers_analyzed"] += circuit.layers_analyzed
            
            #         
            for conn in circuit.connections:
                global_stats["connection_types"][conn.connection_type] += 1
                
                #           (  ,      )
                global_connection_counts[conn.source] += 1
                global_connection_counts[conn.target] += 1
                
            #        (       )
            del circuit
            
        except Exception as e:
            logger.error(f"   Error analyzing {filename}: {e}")
            
    #            
    logger.info("  Calculating global hub neurons...")
    sorted_neurons = sorted(global_connection_counts.items(), key=lambda x: -x[1])
    top_hubs = [n for n, count in sorted_neurons[:20]]
    
    print("\n" + "="*60)
    print(f"GIANT MODEL ANATOMY REPORT: {os.path.basename(model_dir)}")
    print("="*60)
    print(f"   Shards Processed: {len(files)}")
    print(f"   Total Parameters: {global_stats['total_params']:,}")
    print(f"   Layers Analyzed: {global_stats['layers_analyzed']}")
    print(f"   Strong Connections: {global_stats['strong_connections']:,}")
    print(f"   Connection Types: {dict(global_stats['connection_types'])}")
    print("-" * 60)
    print(f"   Top 20 Global Hub Neurons (The Elders):")
    print(f"      {top_hubs}")
    print("="*60)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python topology_batch_runner.py <model_directory>")
        sys.exit(1)
        
    run_batch_analysis(sys.argv[1])
