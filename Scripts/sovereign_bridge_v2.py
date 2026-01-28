"""
SOVEREIGN BRIDGE V2: STANDALONE NEURAL SWITCHBOARD
==================================================
Scripts/sovereign_bridge_v2.py

This script bypasses the broken static import chain by using a standalone 
dynamic loader. It directly initializes ProprioceptionNerve to map the 7D 
fractal body and then resolves components by their functional IDs.
"""

import os
import sys
import logging
import re
import importlib.util
from pathlib import Path
from typing import Dict, Any, Optional, List

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
logger = logging.getLogger("BridgeV2")

# 1. Standalone Proprioception Logic (Bypassing broken imports)
class StandaloneProprioception:
    def __init__(self, root_path: str = "c:/Elysia/Core"):
        self.root_path = Path(root_path)
        self.organ_map: Dict[str, str] = {}

    def scan(self):
        logger.info(f"⚡ [BRIDGE_V2] Scanning fractal body at {self.root_path}...")
        for root, _, files in os.walk(self.root_path):
            for file in files:
                if not file.endswith(".py"): continue
                full_path = Path(root) / file
                try:
                    with open(full_path, "r", encoding="utf-8") as f:
                        content = f.read()
                    
                    # Identify by Decorators
                    cells = re.findall(r'@cell\(\s*["\']([^"\']+)["\']', content)
                    organs = re.findall(r'@organ\(\s*["\']([^"\']+)["\']', content)
                    nodes = re.findall(r'@hyper_node\(\s*["\']([^"\']+)["\']', content)

                    for name in cells: self.organ_map[f"Cell:{name}"] = str(full_path)
                    for name in organs: self.organ_map[f"Organ:{name}"] = str(full_path)
                    for name in nodes: self.organ_map[f"Node:{name}"] = str(full_path)
                except: continue
        logger.info(f"✨ [BRIDGE_V2] {len(self.organ_map)} units registered.")

    def get_module(self, unit_id: str):
        path = self.organ_map.get(unit_id)
        if not path: return None
        
        module_name = f"dynamic_v2_{unit_id.replace(':', '_').lower()}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            # CAUTION: This will still fail if the module HAS static imports to the old paths!
            # But it allows us to at least 'locate' the file.
            return module
        return None

# Singleton-ish
bridge = StandaloneProprioception()
bridge.scan()

def resolve_path(unit_id: str) -> Optional[str]:
    return bridge.organ_map.get(unit_id)

if __name__ == "__main__":
    # Test Resolution
    test_node = "Node:TestRotor"
    path = resolve_path(test_node)
    if path:
        print(f"✅ Found {test_node} at: {path}")
    else:
        print(f"❌ Could not find {test_node}")
        print("Available Nodes:", [k for k in bridge.organ_map.keys() if k.startswith("Node:")])
