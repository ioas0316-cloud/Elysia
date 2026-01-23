import os
import hashlib
import numpy as np
import logging
from typing import Dict, List, Any

logger = logging.getLogger("CorpusScanner")

class CorpusScanner:
    """
    [SELF-INTROSPECTION ORGAN]
    Scans the entire inner universe (File System) of Elysia.
    Maps every file, code, and document into the HyperCosmos.
    """
    
    def __init__(self, root_paths: List[str] = None):
        if root_paths is None:
            # [OMNI-SENSORY] Universal Body Roots
            self.root_paths = [
                "c:\\Elysia",       # Core
                "c:\\Archive",      # Deep Memory
                "c:\\Game",         # External Assets
                "c:\\elysia_seed"   # Ancestral DNA
            ]
        else:
            self.root_paths = root_paths
            
        # [SHELL DEFINITIONS] Assign frequency layers to roots
        self.shell_offsets = {
            "Elysia": 0.0,       # Core Logic
            "Archive": 0.3,      # Depth/History
            "Game": 0.6,         # Physical Volume
            "elysia_seed": -0.2  # Foundation/DNA
        }
        
    def scan_body(self) -> Dict[str, Any]:
        """
        Recursively scans all root paths and returns a topological map.
        Implements [Topological Layering] for organized perception.
        """
        logger.info(f"  Starting Multi-Root Shell Scan of {self.root_paths}...")
        
        body_map = {
            "total_files": 0,
            "total_size": 0,
            "structure": {},
            "qualia_points": []
        }
        
        for root_path in self.root_paths:
            if not os.path.exists(root_path):
                continue
                
            shell_name = os.path.basename(root_path.strip("\\"))
            offset = self.shell_offsets.get(shell_name, 0.5)
            
            logger.info(f"   Scanning Shell Layer: {shell_name} (Offset: {offset})")
            
            for root, dirs, files in os.walk(root_path):
                # 1. Calculate Depth for Layering
                rel_path = os.path.relpath(root, root_path)
                depth = 0 if rel_path == "." else len(rel_path.split(os.sep))
                
                # Filter large/noisy dirs
                dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ["__pycache__", "Weights"]]
                
                for file in files:
                    if file.startswith("."): continue
                    full_path = os.path.join(root, file)
                    try:
                        size = os.path.getsize(full_path)
                    except OSError: continue
                    
                    # 2. Sense Qualia with Layering
                    # Depth makes the qualia more "Solid/Dense" (lower abstraction)
                    signature = self._sense_file_qualia(full_path, file, offset, depth)
                    
                    body_map["total_files"] += 1
                    body_map["total_size"] += size
                    body_map["qualia_points"].append({
                        "path": full_path,
                        "shell": shell_name,
                        "depth": depth,
                        "qualia": signature
                    })
                
        logger.info(f"  Omni-Scan Complete. Shells Organized.")
        return body_map

    def _sense_file_qualia(self, path: str, filename: str, shell_offset: float, depth: int) -> np.ndarray:
        """
        [TOPOLOGICAL SENSING]
        Converts a file into a 7D Qualia Vector with Layered Context.
        """
        # Base from filename
        hash_val = int(hashlib.sha256(filename.encode()).hexdigest(), 16)
        np.random.seed(hash_val % (2**32))
        qualia = np.random.rand(7)
        
        # Apply Shell Offset (Axis 6: Mystery/Origin)
        qualia[6] = np.clip(qualia[6] + shell_offset, 0.0, 1.0)
        
        # Apply Depth (Axis 3: Abstraction)
        # Deeper files are less abstract, more concrete
        qualia[3] = np.clip(qualia[3] - (depth * 0.1), 0.1, 1.0)
        
        # Semantic shifts
        ext = filename.split(".")[-1].lower()
        if ext == "py": qualia[0] += 0.4
        elif ext == "md": qualia[3] += 0.5
        elif ext == "vrm" or ext == "fbx": qualia[1] += 0.6 # Aesthetic/Form
            
        # Normalize
        norm = np.linalg.norm(qualia)
        if norm > 0: qualia /= norm
        return qualia