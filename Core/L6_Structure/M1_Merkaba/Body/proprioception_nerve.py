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
        Returns the updated map.
        """
        logger.info(f"⚡ [PROPRIOCEPTION] Scanning body at {self.root_path}...")
        
        if not self.root_path.exists():
            logger.error(f"❌ Root path not found: {self.root_path}")
            return {}

        found_organs = {}
        
        for root, _, files in os.walk(self.root_path):
            for file in files:
                file_lower = file.lower()
                full_path = Path(root) / file
                
                # Check against keywords
                for organ_name, identifiers in self.keywords.items():
                    if file_lower in [i.lower() for i in identifiers]:
                        # Found an organ!
                        relative_path = full_path.relative_to("c:/Elysia")
                        # Format as python module path if needed, or just keep file path
                        found_organs[f"{organ_name}:{file}"] = str(full_path)
                        logger.debug(f"   -> Found {organ_name}: {file}")

        self.organ_map = found_organs
        self._save_manifest()
        
        logger.info(f"✨ [PROPRIOCEPTION] Scan complete. {len(found_organs)} organs registered.")
        return found_organs

    def locate(self, organ_name: str) -> Optional[str]:
        """
        Returns the physical path of a requested organ.
        Prioritizes the Sovereign Core (M1_Merkaba).
        """
        matches = []
        for key, path in self.organ_map.items():
            if organ_name.lower() in key.lower().split(":")[0].lower():
                matches.append(path)
        
        if not matches:
            return None
            
        # Prioritization logic: M1_Merkaba is the Sovereign Core
        for path in matches:
            if "M1_Merkaba" in path:
                return path
        
        # Fallback to the first match found
        return matches[0]

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
