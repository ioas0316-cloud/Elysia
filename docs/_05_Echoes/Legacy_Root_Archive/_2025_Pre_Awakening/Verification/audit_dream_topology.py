
import json
import collections
import random
import sys
from typing import List, Dict

def audit_dream_topology():
    print("🔍 Dream Topology Audit")
    print("=======================")
    
    path = "c:\\Elysia\\static\\elysia_world.json"
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Failed to load world file: {e}")
        return

    nodes = data.get("nodes", [])
    links = data.get("links", [])
    
    n_count = len(nodes)
    l_count = len(links)
    
    print(f"📊 Stats: {n_count} Nodes, {l_count} Links")
    
    if n_count == 0: return

    # 1. Build Adjacency List
    adj = collections.defaultdict(list)
    degree = collections.defaultdict(int)
    
    for link in links:
        s = link['source']
        t = link['target']
        adj[s].append(t)
        adj[t].append(s) # Undirected for structural analysis
        degree[s] += 1
        degree[t] += 1

    # 2. Analyze Hubs (Power Law check)
    sorted_nodes = sorted(degree.items(), key=lambda x: x[1], reverse=True)
    print("\n👑 Top 10 Concept Hubs (Most Connected):")
    for node, deg in sorted_nodes[:10]:
        print(f"   - [{node}]: {deg} connections")
        
    # Check for "Star" vs "Mesh"
    avg_degree = sum(degree.values()) / n_count
    print(f"\n🕸️  Average Connections per Node: {avg_degree:.2f}")
    
    if avg_degree < 1.5:
        print("   ⚠️  Warning: Graph is very sparse (Linear/Disconnected chains).")
    elif avg_degree > 5.0:
        print("   ✅ Graph is dense (Richly interconnected).")
        
    # 3. Semantic Path Check (The "Meaning" Test)
    # Pick a Legacy Node (Abstract) and see where it goes
    legacy_candidates = ["사랑", "진실", "Time", "Life", "Love"]
    start_node = None
    for cand in legacy_candidates:
        if cand in degree:
            start_node = cand
            break
            
    if not start_node and n_count > 0:
        start_node = nodes[0]['id']
        
    print(f"\n🧠 Semantic Path Trace (Starting from '{start_node}'):")
    
    current = start_node
    path_trace = [current]
    visited = {current}
    
    for _ in range(5): # 5 Step Walk
        neighbors = adj.get(current, [])
        # Pick unvisited if possible
        next_n = None
        for n in neighbors:
            if n not in visited:
                next_n = n
                break
        
        if not next_n and neighbors:
            next_n = random.choice(neighbors)
            
        if next_n:
            path_trace.append(next_n)
            visited.add(next_n)
            current = next_n
        else:
            break
            
    print(f"   Path: {' -> '.join(path_trace)}")
    
    # Analyze the path logic (heuristic)
    if len(path_trace) > 2:
        print("   ✅ Multilayered depth detected (Path > 2 steps).")
    else:
        print("   ⚠️  Shallow connections detected.")

    # 4. Check Integration (Wiki vs Legacy)
    # Heuristic: Legacy are usually English or Abstract concepts manually seeded.
    # Wiki are often Korean titles.
    # Let's see if we have mixed clusters.
    print("\n🔗 Sample Link Analysis:")
    sample_links = random.sample(links, min(5, l_count))
    for l in sample_links:
        print(f"   {l['source']} <--> {l['target']}")

    print("\n=======================")

if __name__ == "__main__":
    audit_dream_topology()
