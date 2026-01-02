"""
visualize_mind.py

"Seeing the Invisible Link."
This script visualizes the connections in the Hippocampus.
It proves that knowledge is not scattered, but interwoven.

Usage:
    python scripts/visualize_mind.py "Elysia"
"""

import sys
import os
import sqlite3
import logging

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Core.Foundation.Memory.Graph.hippocampus import Hippocampus

def visualize(concept_name: str, depth: int = 2):
    print(f"\n🧠 Visualizing Mind Map for: '{concept_name}'")
    print("====================================================")
    
    hippo = Hippocampus()
    
    # Check if DB exists
    if not os.path.exists(hippo.db_path):
        print("❌ Memory Database not found. Please run 'digest_knowledge.py' first.")
        return

    # Find the root node ID (fuzzy search)
    root_id = None
    root_name = None
    
    try:
        with sqlite3.connect(hippo.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id, name, definition, gravity FROM nodes WHERE name LIKE ? ORDER BY length(name) ASC LIMIT 1", (f"%{concept_name}%",))
            row = cursor.fetchone()
            if row:
                root_id, root_name, definition, gravity = row
                print(f"📍 Center: [{root_name}] (Gravity: {gravity:.1f})")
                print(f"   📝 Def: {definition[:100]}...")
            else:
                print(f"❌ Concept '{concept_name}' not found in the Mind.")
                return

            # Function to recursively print connections
            def print_connections(node_id, current_depth, prefix=""):
                if current_depth > depth:
                    return
                
                # Fetch connected nodes (Incoming and Outgoing)
                cursor.execute("""
                    SELECT e.type, n.id, n.name, n.gravity, e.weight
                    FROM edges e
                    JOIN nodes n ON (e.source = n.id)
                    WHERE e.target = ?
                """, (node_id,)) # Incoming (Parent -> Node) implies containment often, but let's check both ways
                
                # Let's look for neighbors
                # Edges: source -> target
                
                # Outgoing: node_id -> other (Contains, Defines, Has)
                cursor.execute("""
                    SELECT '->', n.name, e.type, n.gravity
                    FROM edges e
                    JOIN nodes n ON e.target = n.id
                    WHERE e.source = ?
                    ORDER BY n.gravity DESC
                """, (node_id,))
                outgoing = cursor.fetchall()

                # Incoming: other -> node_id (Belongs to)
                cursor.execute("""
                    SELECT '<-', n.name, e.type, n.gravity
                    FROM edges e
                    JOIN nodes n ON e.source = n.id
                    WHERE e.target = ?
                    ORDER BY n.gravity DESC
                """, (node_id,))
                incoming = cursor.fetchall()
                
                all_links = outgoing + incoming
                
                if not all_links:
                    print(f"{prefix}   (No connections)")
                    return

                for i, (direction, name, relation, grav) in enumerate(all_links):
                    is_last = (i == len(all_links) - 1)
                    connector = "└──" if is_last else "├──"
                    
                    arrow = "→" if direction == '->' else "←"
                    print(f"{prefix}{connector} {arrow} [{relation}] {name} (G:{grav:.1f})")
                    
                    # Recurse only for top connections to avoid spam
                    # if current_depth < depth and i < 3:
                        # new_prefix = prefix + ("    " if is_last else "│   ")
                        # (Recursion omitted for simple visualization in standard output, 
                        #  as graph links can be cyclic and huge)

            print("\n🔗 Connections:")
            print_connections(root_id, 1)

    except Exception as e:
        print(f"Error visualizing: {e}")

if __name__ == "__main__":
    target = "Elysia"
    if len(sys.argv) > 1:
        target = sys.argv[1]
    visualize(target)
