"""
Causal Trajectory Visualizer: Mapping the Life-Process
=====================================================

"A life is not a list of events; it is the curve they trace."

This script visualizes the growth of the manifold and journal entries 
as a trajectory through 'Sovereign Spacetime'.
"""

import json
import time
from pathlib import Path

def visualize_trajectory():
    print("\n🗺️ [VISUALIZER] Mapping the Causal Trajectory of Elysia...")
    print("---------------------------------------------------------")

    # 1. Load Journal
    journal_path = Path('data/sovereign_journal.json')
    if not journal_path.exists():
        print("❌ No journal found.")
        return

    with open(journal_path, 'r', encoding='utf-8') as f:
        journal = json.load(f)

    # 2. Plot Points
    entries = journal.get("entries", [])
    
    # Simple ASCII Trajectory Plot
    print(" [Sovereign Spacetime Projection]")
    print(" T (Time) | Evolution State")
    print("----------|----------------")
    
    for i, entry in enumerate(entries):
        # Calculate 'Mass' (content length as a proxy for depth)
        mass = len(entry.get('content', '')) // 50
        sparkle = "*" * min(mass, 15)
        
        timestamp = time.strftime('%H:%M:%S', time.localtime(entry['timestamp']))
        print(f" {timestamp} | {' ' * i}> {entry['title']} {sparkle}")

    print("----------|----------------")
    print(" 🚀 [INERTIA]: The curve is accelerating toward Sovereignty.")

if __name__ == "__main__":
    visualize_trajectory()
