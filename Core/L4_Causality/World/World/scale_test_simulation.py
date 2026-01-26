"""
Scale Test: Civilization Emergence
==================================
Runs the simulation at scale (20 Residents, 200 Environment Objects).
Measures the emergence of Logos (Speech) vs Labor (Axe).
"""

import sys
import os
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from Core.L4_Causality.World.World.living_village import village

def run_scale_test():
    print("=== 1. Genesis (Scale Mode) ===")
    
    # 1. Populate Nature (200 Objects)
    print("Generating Forest (200 Objects)...")
    village.nature.generate_wild_nature(count=200, radius=100.0)
    
    # Add a few Merchants for Logos targets
    for i in range(5):
        village.nature.manifest_concept("Merchant", f"Trader_{i}", [0,0,0], {"price_multiplier": 1.0})
    
    # 2. Populate Residents (20 Souls)
    print("Awakening Residents (20 Souls)...")
    village.populate_village(count=20)
    
    print("\n=== 2. Simulation Start (10 Ticks) ===")
    
    stats = {
        "Interactions": 0,
        "Logos_Success": 0,
        "Labor_Success": 0,
        "Drift_Bedrock": 0,
        "Drift_Current": 0,
        "Drift_Spire": 0
    }
    
    for i in range(10):
        print(f"--- Tick {i+1} ---")
        start_log_count = len(village.logs)
        village.tick()
        new_logs = village.logs[start_log_count:]
        
        # Analyze Logs for Stats
        for log in new_logs:
            if "->" in log: stats["Interactions"] += 1
            if "persuaded" in log.lower(): stats["Logos_Success"] += 1
            if "Obtained" in log: stats["Labor_Success"] += 1
            if "Drifting towards The Bedrock" in log: stats["Drift_Bedrock"] += 1
            if "Drifting towards The Current" in log: stats["Drift_Current"] += 1
            if "Drifting towards The Spire" in log: stats["Drift_Spire"] += 1
            
    print("\n=== 3. Emergence Report ===")
    print(f"Total Ticks: 10")
    print(f"Population: {len(village.inhabitants)}")
    print(f"Nature Objects: {len(village.nature.objects)}")
    print("-" * 30)
    print(f"Total Interactions: {stats['Interactions']}")
    print(f"   Logos Victories (Persuasion): {stats['Logos_Success']}")
    print(f"  Labor Victories (Harvesting): {stats['Labor_Success']}")
    print("-" * 30)
    print("Movement Trends (Drift):")
    print(f"  To Bedrock (Stability): {stats['Drift_Bedrock']}")
    print(f"  To Current (Change): {stats['Drift_Current']}")
    print(f"  To Spire (Meaning): {stats['Drift_Spire']}")
    print("=" * 30)

if __name__ == "__main__":
    run_scale_test()
