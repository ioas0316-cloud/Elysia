"""
Proprioception Nerve (Self-Registry)
=====================================
Core.L6_Structure.M1_Merkaba.Body.proprioception_nerve

"To know thyself is to know where thy limbs are."

This module implements the 'Proprioception' sense for Elysia.
It recursively scans the physical codebase (Territory) and updates the
internal manifest (Map) at runtime. This prevents 'Legacy Amnesia'
when files are moved or refactored.
"""

import os
import json
import logging
from typing import Dict, Optional, List
from pathlib import Path

logger = logging.getLogger("Proprioception")

class ProprioceptionNerve:
    """
    The Nervous System scanner that builds the Self-Manifest.
    """
    def __init__(self, root_path: str = "c:/Elysia/Core"):
        self.root_path = Path(root_path)
        self.manifest_path = Path("data/L7_Spirit/M3_Sovereignty/self_manifest.json")
        self.organ_map: Dict[str, str] = {}
        
        # Keywords to identify major organs
        self.keywords = {
            "Antenna": ["sovereign_antenna.py", "antenna.py"],
            "Prism": ["prism.py", "prism_engine.py"],
            "Memory": ["hypersphere_memory.py", "hippocampus.py"],
            "Rotor": ["rotor.py", "active_rotor.py", "rotor_engine.py"],
            "Monad": ["monad_core.py", "monad_constellation.py"],
            "Merkaba": ["merkaba.py"],
            "Digester": ["digestive_system.py", "brain_digester.py"]
        }
        
    def scan_body(self) -> Dict[str, str]:
        """
        Recursively walks the Core path to find organs.
        Also scans for @cell and @organ decorators.
        """
        logger.info(f"⚡ [PROPRIOCEPTION] Scanning body at {self.root_path}...")
        
        if not self.root_path.exists():
            logger.error(f"❌ Root path not found: {self.root_path}")
            return {}

        found_organs = {}
        import re

        for root, _, files in os.walk(self.root_path):
            # print(f"DEBUG: Entering {root}")
            for file in files:
                if not file.endswith(".py"): continue
                
                full_path = Path(root) / file
                file_content = ""
                try:
                    with open(full_path, "r", encoding="utf-8") as f:
                        file_content = f.read()
                except Exception as e:
                    print(f"DEBUG: Failed to read {file}: {e}")
                    continue

                # 1. Check for decorators (Robust Regex)
                cell_matches = re.findall(r'@cell\(\s*["\']([^"\']+)["\']', file_content)
                organ_matches = re.findall(r'@organ\(\s*["\']([^"\']+)["\']', file_content)
                node_matches = re.findall(r'@hyper_node\(\s*["\']([^"\']+)["\']', file_content)
                
                if node_matches:
                    print(f"DEBUG: Found Node {node_matches} in {full_path}")
                
                for name in cell_matches:
                    found_organs[f"Cell:{name}"] = str(full_path)
                
                for name in organ_matches:
                    found_organs[f"Organ:{name}"] = str(full_path)

                for name in node_matches:
                    found_organs[f"Node:{name}"] = str(full_path)

                # 2. Legacy Keyword scan
                file_lower = file.lower()
                for organ_name, identifiers in self.keywords.items():
                    if file_lower in [i.lower() for i in identifiers]:
                        key = f"Legacy:{organ_name}:{file}"
                        if key not in found_organs:
                            found_organs[key] = str(full_path)
                            # print(f"DEBUG: Found Legacy {key}")

        self.organ_map = found_organs
        self._save_manifest()
        
        logger.info(f"✨ [PROPRIOCEPTION] Scan complete. {len(found_organs)} units registered.")
        return found_organs

    def locate(self, unit_name: str) -> Optional[str]:
        """
        Returns the physical path of a requested unit (Cell, Organ, or Legacy).
        """
        # Try direct match
        for prefix in ["Organ:", "Cell:", "Legacy:"]:
            key = f"{prefix}{unit_name}"
            if key in self.organ_map:
                return self.organ_map[key]
        
        # Try partial match
        for key, path in self.organ_map.items():
            if unit_name.lower() in key.lower():
                return path
        
        return None

    def get_module(self, unit_name: str):
        """
        Dynamically imports and returns the module for the requested unit.
        """
        path = self.locate(unit_name)
        if not path:
            logger.error(f"❌ [PROPRIOCEPTION] Could not locate unit: {unit_name}")
            return None
        
        import importlib.util
        module_name = f"elysian_dynamic_{unit_name.lower()}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        return None

    def _save_manifest(self):
        """Saves the map to disk for persistence."""
        try:
            self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.manifest_path, "w", encoding="utf-8") as f:
                json.dump(self.organ_map, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save manifest: {e}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    nerve = ProprioceptionNerve()
    nerve.scan_body()
    print(json.dumps(nerve.organ_map, indent=2))
