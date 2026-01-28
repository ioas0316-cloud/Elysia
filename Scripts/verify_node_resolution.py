import logging
import sys
import os

# Ensure root is in path
root = "c:/Elysia"
if root not in sys.path:
    sys.path.insert(0, root)

from Core.L6_Structure.M5_Engine.sovereign_bridge import bridge, resolve

def test_node_resolution():
    print("🌉 [NODE_TEST] Scanning for HyperNodes...")
    bridge.nerve.scan_body()
    
    # 1. Resolve by ID using the new Node Connector
    print("   -> Attempting to resolve Node:TestRotor")
    node_cls = resolve("Node:TestRotor")
    
    if node_cls:
        print("✅ Success! Resolved HyperNode.")
        # If it's a class, instantiate and call
        if isinstance(node_cls, type):
            instance = node_cls()
            print(f"   -> Result: {instance.spin_test()}")
        else:
            print(f"   -> Resolved as module: {node_cls}")
    else:
        print("❌ Failed to resolve HyperNode.")
        # Debug: list keys that start with Node:
        node_keys = [k for k in bridge.nerve.organ_map.keys() if k.startswith("Node:")]
        print(f"   -> Available Nodes: {node_keys}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_node_resolution()
