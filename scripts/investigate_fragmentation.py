"""
Script: Investigate Knowledge Fragmentation
===========================================

Counts nodes in the major knowledge stores to verify user's claim of ~30k nodes.
Check:
1. elysia_rainbow.json (336MB)
2. memory.db (719MB)
3. brain_state.pt
4. InternalUniverse (Current)
"""

import json
import sqlite3
import torch
import os
import sys

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from Core.01_Foundation.05_Foundation_Base.Foundation.internal_universe import InternalUniverse

DATA_DIR = "c:\\Elysia\\data"

def check_rainbow():
    path = os.path.join(DATA_DIR, "elysia_rainbow.json")
    if not os.path.exists(path):
        return 0, "Not Found"
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            # Assuming list or dict structure
            if isinstance(data, list):
                return len(data), "List Items"
            elif isinstance(data, dict):
                # Check for 'nodes' key or just keys
                if 'nodes' in data:
                    return len(data['nodes']), "Nodes in Dict"
                return len(data), "Keys in Dict"
            return 0, "Unknown Structure"
    except Exception as e:
        return 0, f"Error: {e}"

def check_db():
    path = os.path.join(DATA_DIR, "memory.db")
    if not os.path.exists(path):
        return 0, "Not Found"
    
    try:
        conn = sqlite3.connect(path)
        cursor = conn.cursor()
        
        # Check tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        total_rows = 0
        details = []
        
        for table in tables:
            t_name = table[0]
            cursor.execute(f"SELECT COUNT(*) FROM {t_name}")
            count = cursor.fetchone()[0]
            total_rows += count
            details.append(f"{t_name}:{count}")
            
        conn.close()
        return total_rows, ", ".join(details)
    except Exception as e:
        return 0, f"Error: {e}"

def check_brain_state():
    path = os.path.join(DATA_DIR, "brain_state.pt")
    if not os.path.exists(path):
        return 0, "Not Found"
    
    try:
        state = torch.load(path)
        if 'id_to_idx' in state:
            return len(state['id_to_idx']), "Torch Nodes"
        return 0, "Unknown State"
    except Exception as e:
        return 0, f"Error: {e}"

def check_internal_universe():
    try:
        u = InternalUniverse()
        return len(u.coordinate_map), "Concepts in Memory"
    except:
        return 0, "Failed to Init"

def main():
    print("🕵️ Investigating Knowledge Fragmentation...")
    print("===========================================")
    
    # 1. Rainbow JSON
    count, msg = check_rainbow()
    print(f"🌈 elysia_rainbow.json: {count} nodes ({msg})")
    
    # 2. SQLite DB
    count, msg = check_db()
    print(f"🗄️ memory.db:          {count} rows ({msg})")
    
    # 3. Torch Brain State
    count, msg = check_brain_state()
    print(f"🧠 brain_state.pt:     {count} nodes ({msg})")
    
    # 4. Current Universe
    count, msg = check_internal_universe()
    print(f"🌌 InternalUniverse:   {count} concepts ({msg})")
    
    print("===========================================")

if __name__ == "__main__":
    main()
