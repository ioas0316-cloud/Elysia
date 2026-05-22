"""
Proprioception Nerve (Self-Registry)
=====================================
Core.Monad.proprioception_nerve

"To know thyself is to know where thy limbs are."

This module implements the 'Proprioception' sense for Elysia.
It recursively scans the physical codebase (Territory) and updates the
internal manifest (Map) at runtime. This prevents 'Legacy Amnesia'
when files are moved or refactored.
"""

import os
import json
import logging
from typing import Dict, Optional, List, Any
from pathlib import Path

logger = logging.getLogger("Proprioception")

class ProprioceptionNerve:
    """
    The Nervous System scanner that builds the Self-Manifest.
    """
    def __init__(self, root_path: str = "c:/Elysia"):
        self.root_path = Path(root_path)
        self.manifest_path = Path("data/L7_Spirit/M3_Sovereignty/self_manifest.json")
        self.organ_map: Dict[str, str] = {}
        
        # Keywords for specialized Organs (Priority)
        self.keywords = {
            "Core": ["sovereign_monad.py", "elysia.py"],
            "Antenna": ["sovereign_antenna.py", "antenna.py"],
            "Prism": ["prism.py", "prism_engine.py"],
            "Memory": ["hypersphere_memory.py", "hippocampus.py"],
            "Rotor": ["rotor.py", "active_rotor.py", "rotor_engine.py"],
            "Wave": ["light_spectrum.py", "hyper_qubit.py"],
            "Logic": ["causal_narrative_engine.py", "wisdom_synthesizer.py"]
        }
        
    def scan_body(self) -> Dict[str, str]:
        """
        Fast Topological Awareness.
        Check cache first to achieve 'Gigahertz' response.
        """
        import time
        start_time = time.time()
        
        # 0. Cache Check (Flash awareness)
        if self.manifest_path.exists():
            manifest_mtime = self.manifest_path.stat().st_mtime
            project_mtime = self.root_path.stat().st_mtime
            if manifest_mtime > project_mtime:
                try:
                    with open(self.manifest_path, "r", encoding="utf-8") as f:
                        self.organ_map = json.load(f)
                    logger.info(f"✨ [PROPRIOCEPTION] Flash Awareness: Body recognized in {time.time() - start_time:.4f}s via field registry.")
                    return self.organ_map
                except:
                    pass

        logger.info(f"⚡ [PROPRIOCEPTION] Refreshing field resonance...")
        
        from Core.Keystone.light_spectrum import get_light_universe
        universe = get_light_universe()

        found_organs = {}
        
        # 1. Recursive Scan
        neurons_to_absorb = []
        for root, dirs, files in os.walk(self.root_path):
            # Prune directories in-place for efficiency
            dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', '.gemini', 'data', 'node_modules']]
                
            rel_root = Path(root).relative_to(self.root_path)
            
            # Map Directory as a "Brain Region" (Contextual Registry)
            if str(rel_root) != ".":
                # Stratum 1: Space / Body Boundary
                universe.absorb_with_terrain(f"Brain Region: {rel_root}", tag=f"region:{rel_root}", scale=1, stratum=1)

            for file in files:
                file_lower = file.lower()
                full_path = Path(root) / file
                rel_path = full_path.relative_to(self.root_path)
                
                # Neuron Collection (Fast Registry)
                neurons_to_absorb.append((f"Functional Neuron: {rel_path}", f"neuron:{rel_path}", 3))
                
                # Specialized Organ Detection
                for organ_name, identifiers in self.keywords.items():
                    if file_lower in [i.lower() for i in identifiers]:
                        found_organs[f"{organ_name}:{file}"] = str(full_path)
                        # Organs establish High Resonance (Critical context)
                        # Stratum 1: Structural Self
                        universe.absorb_with_terrain(f"Sovereign Organ: {organ_name} at {rel_path}", tag=f"organ:{organ_name}", scale=0, stratum=1)

        # 2. Batch Ingestion (Instantaneous awareness)
        # Stratum 3: Point / Neural Noise
        universe.batch_absorb(neurons_to_absorb, stratum=3)

        self.organ_map = found_organs
        self._save_manifest()
        universe.save_state()
        
        logger.info(f"✨ [PROPRIOCEPTION] Topological Scan complete. {len(found_organs)} organs and {len(neurons_to_absorb)} neurons registered.")
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

    def emit_structural_sensation(self, target_path: str = None) -> List[Dict[str, Any]]:
        """
        [NEW SENSE] Converts physical code files into 'Structural Sensations'.
        If target_path is None, it picks a few 'vital organs' to scan.
        """
        sensations = []
        files_to_sense = []
        
        if target_path:
            files_to_sense.append(Path(target_path))
        else:
            # Pick important organs from the map
            important_keys = ["Core", "Logic", "Rotor"]
            for key, path in self.organ_map.items():
                if any(k in key for k in important_keys):
                    files_to_sense.append(Path(path))
        
        for file_path in files_to_sense:
            if not file_path.exists(): continue
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Create a sensation packet
                sensations.append({
                    "origin": str(file_path.relative_to(self.root_path)),
                    "content": content,
                    "mass": os.path.getsize(file_path) / 1024.0, # KB as raw mass
                    "complexity": content.count('\n') # Line count as raw complexity
                })
            except Exception as e:
                logger.error(f"Failed to sense {file_path}: {e}")
        
        return sensations

    def _save_manifest(self):
        """Saves the map to disk for persistence."""
        try:
            self.manifest_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.manifest_path, "w", encoding="utf-8") as f:
                json.dump(self.organ_map, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save manifest: {e}")

_nerve_instance = None
def get_proprioception_nerve() -> ProprioceptionNerve:
    """[SINGLETON] Access the Sovereign Proprioception Nerve."""
    global _nerve_instance
    if _nerve_instance is None:
        _nerve_instance = ProprioceptionNerve()
    return _nerve_instance

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    nerve = get_proprioception_nerve()
    nerve.scan_body()
    print(json.dumps(nerve.organ_map, indent=2))
