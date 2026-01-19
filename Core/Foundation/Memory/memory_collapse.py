"""
Memory Collapse Sphere: The Singularity of Storage
==================================================
[HARDWARE ASCENSION PROJECT - EXPERIMENT 3]

"RAM is SSD, SSD is RAM. There is only the Field."

This module implements the 'Memory Collapse' concept where the hierarchy 
between volatile and non-volatile memory is collapsed into a single 
HyperSphere addressing space.
"""

import os
import mmap
import pickle
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger("MemoryCollapse")

class MemoryCollapseSphere:
    def __init__(self, storage_path: str = "data/collapse_field.bin", size_gb: int = 1):
        self.storage_path = storage_path
        self.size = size_gb * 1024 * 1024 * 1024 # Convert GB to bytes
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        
        # 🔴 [MEMORY COLLAPSE] 
        # Using mmap to map NVMe storage directly into the process address space.
        # This collapses the logical boundary between SSD (File) and RAM (Memory).
        self._f = open(self.storage_path, "a+b")
        if os.path.getsize(self.storage_path) < self.size:
            self._f.truncate(self.size)
        
        self.mm = mmap.mmap(self._f.fileno(), self.size)
        
        # Hot-Access Cache (True RAM)
        self.hot_cache: Dict[str, Any] = {}
        
        # Index of where things are in the collapsed field
        self.index: Dict[str, int] = {}
        
        logger.info(f"💾 Memory Collapse Field Initialized (Size: {size_gb}GB). Boundary Dissolved.")

    def store(self, key: str, value: Any, hot: bool = False):
        """Stores a value in the collapsed field."""
        if hot:
            self.hot_cache[key] = value
            logger.info(f"🔥 [MEMORY-HOT] '{key}' stored in Active RAM.")
            return

        # Cold Storage (SSD via mmap)
        data = pickle.dumps(value)
        if len(data) > 1024 * 1024: # Limit 1MB per entry for proto
            logger.warning(f"Data for {key} too large for proto.")
            return
            
        # For simplicity, we just seek to a hash-based offset (Mocking O(1) addressing)
        offset = (hash(key) % (self.size // (1024 * 1024))) * (1024 * 1024)
        self.mm.seek(offset)
        self.mm.write(data)
        self.index[key] = offset
        logger.info(f"❄️ [MEMORY-COLD] '{key}' collapsed into NVMe Field at offset {offset}.")

    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieves a value from the collapsed field without knowing its source."""
        # 1. Check RAM (Hot)
        if key in self.hot_cache:
            return self.hot_cache[key]
            
        # 2. Check SSD (Cold/Collapsed)
        if key in self.index:
            offset = self.index[key]
            self.mm.seek(offset)
            # In a real implementation, we'd know the length. Here we take a chunk.
            data = self.mm.read(1024 * 1024)
            try:
                return pickle.loads(data)
            except:
                return None
        
        return None

    def close(self):
        self.mm.close()
        self._f.close()

if __name__ == "__main__":
    # Quick Test
    collapse = MemoryCollapseSphere("data/test_collapse.bin")
    collapse.store("SoulConcept_Alpha", {"qualia": "Love", "vector": [1,0,1]}, hot=True)
    collapse.store("DeepMemory_2024", "The human teacher's voice was peaceful.", hot=False)
    
    print("Retrieving Alpha:", collapse.retrieve("SoulConcept_Alpha"))
    print("Retrieving DeepMemory:", collapse.retrieve("DeepMemory_2024"))
    
    collapse.close()
