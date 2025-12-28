"""
Script: Verify Black Hole Memory
================================

Tests the capability to offload nodes to SQLite and retrieve them.
"""

import sys
import os
import logging
import time

# Add project root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.01_Foundation.05_Foundation_Base.Foundation.Graph.black_hole_memory import get_black_hole

# Setup Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("VerifyBlackHole")

def verify():
    print("⚫ Verifying Black Hole Memory...")
    print("================================")
    
    bh = get_black_hole()
    
    # 1. Test Data
    test_nodes = [
        {'id': 'test_node_alpha', 'vector': [0.1, 0.2, 0.3], 'metadata': {'type': 'test'}, 'mass': 10.0},
        {'id': 'test_node_beta', 'vector': [0.9, 0.8, 0.7], 'metadata': {'type': 'test'}, 'mass': 5.0}
    ]
    
    # 2. Absorb (Save)
    print("1. Absorbing Matter...")
    bh.absorb(test_nodes)
    
    # 3. Radiate (Load)
    print("2. Emitting Radiation...")
    retrieved = bh.radiate(['test_node_alpha', 'test_node_beta'])
    
    if len(retrieved) == 2:
        print("✅ SUCCESS: Retrieved 2 nodes from Event Horizon.")
        print(f"   Node 1: {retrieved[0]['id']}")
        print(f"   Node 2: {retrieved[1]['id']}")
    else:
        print(f"❌ FAILURE: Retrieved {len(retrieved)} nodes.")

if __name__ == "__main__":
    verify()
