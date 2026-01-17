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
            self.root_paths = ["c:\\Elysia", "c:\\Archive"]
        else:
            self.root_paths = root_paths
        
    def scan_body(self) -> Dict[str, Any]:
        """
        Recursively scans all root paths and returns a topological map.
        This is equivalent to 'Proprioception' (Feeling one's own body).
        """
        logger.info(f"🧘 Starting Deep Body Scan of {self.root_paths}...")
        
        body_map = {
            "total_files": 0,
            "total_size": 0,
            "structure": {},
            "qualia_points": [] # Points to be imprinted in Prism
        }
        
        for root_path in self.root_paths:
            if not os.path.exists(root_path):
                logger.warning(f"⚠️ Scan Path not found: {root_path}")
                continue
                
            logger.info(f"   Scanning Sector: {root_path}")
            
            for root, dirs, files in os.walk(root_path):
                current_structure = body_map["structure"]
                
                # Navigate to current structure node
                try:
                    rel_path = os.path.relpath(root, root_path)
                except ValueError:
                    rel_path = root # Fallback for different drives
                    
                if rel_path == ".":
                    path_parts = [os.path.basename(root_path)] # Use root folder name as base
                else:
                    path_parts = [os.path.basename(root_path)] + rel_path.split(os.sep)
                
                # Skip hidden directories and large weight/log folders
                dirs[:] = [d for d in dirs if not d.startswith('.')]
                if "Weights" in root or "Logs" in root or "__pycache__" in root:
                    continue
                
                for part in path_parts:
                    current_structure = current_structure.setdefault(part, {})
                
                for file in files:
                    if file.startswith("."):
                        continue
                        
                    full_path = os.path.join(root, file)
                    try:
                        size = os.path.getsize(full_path)
                    except OSError:
                        continue # Skip inaccessible files
                    
                    # Generate a 'Soul Signature' for the file (Hash + Content Analysis)
                    signature = self._sense_file_qualia(full_path, file)
                    
                    # Register to body map
                    current_structure[file] = {
                        "size": size,
                        "qualia": signature.tolist()
                    }
                    
                    body_map["total_files"] += 1
                    body_map["total_size"] += size
                    body_map["qualia_points"].append({
                        "path": full_path, # Use full path for multi-root clarity
                        "qualia": signature
                    })
                
        logger.info(f"🧘 Body Scan Complete. Cells: {body_map['total_files']}, Mass: {body_map['total_size']} bytes")
        return body_map

    def _sense_file_qualia(self, path: str, filename: str) -> np.ndarray:
        """
        Converts a file into a 7D Qualia Vector.
        This determines 'Where' this file lives in the PrismSpace.
        """
        # 1. Structural Hash (Deterministic Base)
        # Using filename to determine base position
        hash_val = int(hashlib.sha256(filename.encode()).hexdigest(), 16)
        np.random.seed(hash_val % (2**32))
        base_qualia = np.random.rand(7)
        
        # 2. Semantic shift based on extension
        ext = filename.split(".")[-1].lower()
        if ext == "py": # Logic / Function
            base_qualia[0] += 0.5 # Logic axis
            base_qualia[5] += 0.5 # Utility axis
        elif ext == "md": # Knowledge / Philosophy
            base_qualia[3] += 0.6 # Abstraction axis
            base_qualia[0] += 0.3 # Logic axis
        elif ext == "json" or ext == "yaml": # Data / Structure
            base_qualia[2] += 0.7 # Precision axis
            
        # 3. Content Analysis (DISABLED FOR SPEED)
        # Reading files is too slow. Rely on filename semantics.
        # try:
        #     with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        #         head = f.read(1024)
        #         if "class " in head: # Object
        #             base_qualia[1] += 0.4 # Creation
        #         if "def " in head: # Action
        #             base_qualia[5] += 0.4 # Utility
        #         if "#" in head: # Comment/Thought
        #             base_qualia[3] += 0.2 # Abstraction
        # except:
        #     pass
            
        # Normalize
        norm = np.linalg.norm(base_qualia)
        if norm > 0:
            base_qualia /= norm
            
        return base_qualia
