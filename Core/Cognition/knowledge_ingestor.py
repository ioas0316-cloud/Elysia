"""
Knowledge Ingestor (The Nervous System)
=======================================
Core.Cognition.knowledge_ingestor

"The Brain must know the Body."

Roles:
1.  **Scanner**: Recursively crawls the file system (`docs/`, `Core/`).
2.  **Reader**: Extracts clean text from Markdown and Python files.
3.  **Feeder**: Sends text to `MeaningDeconstructor` and stores results in `TorchGraph`.

"""

import os
import logging
import glob
from typing import List, Dict
from Core.System.torch_graph import TorchGraph
from Core.Cognition.meaning_deconstructor import MeaningDeconstructor
from Core.Cognition.universal_digestor import RawKnowledgeChunk, ChunkType

logger = logging.getLogger("KnowledgeIngestor")

class KnowledgeIngestor:
    def __init__(self, root_path: str = "c:\\Elysia"):
        self.root_path = root_path
        self.graph = TorchGraph()
        self.deconstructor = MeaningDeconstructor()
        
    def scan_and_ingest(self, target_folders: List[str] = ["docs", "Core"]):
        """
        Main Loop: Scans folders, reads files, seeds the Brain.
        """
        print(f"  [Ingestor] waking up... Scanning: {target_folders}")
        total_files = 0
        total_concepts = 0
        
        for folder in target_folders:
            search_path = os.path.join(self.root_path, folder, "**", "*.*")
            files = glob.glob(search_path, recursive=True)
            
            for file_path in files:
                if self._should_ignore(file_path):
                    continue
                    
                total_files += 1
                self.ingest_file(file_path)
                
        # Save the brain after learning
        self.graph.ignite_gravity() # Forge connections
        self.graph.save_state()
        print(f"  [Ingestor] Digestion Complete. Files: {total_files}")

    def _should_ignore(self, path: str) -> bool:
        """Filters out noise."""
        if "__pycache__" in path: return True
        if ".git" in path: return True
        if path.endswith(".pyc"): return True
        if "Archive" in path and "99_ARCHIVE" not in path: return True # Ignore legacy if strictly needed
        # We actually WANT to read 99_ARCHIVE if it contains history, 
        # but maybe focus on Active Docs first. 
        # For now, let's read everything visible.
        return not (path.endswith(".md") or path.endswith(".py"))

    def ingest_file(self, file_path: str) -> List[RawKnowledgeChunk]:
        """Reads a single file and extracts DNA."""
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                
            relative_path = os.path.relpath(file_path, self.root_path)
            print(f"     Reading: {relative_path}")

            # 1. Create a Node for the File itself
            file_node_id = f"File:{relative_path}"
            self.graph.add_node(
                file_node_id, 
                metadata={"type": "File", "path": file_path, "content_snippet": content[:100]}
            )
            
            # 2. Package as a Chunk for the Stream
            chunk = RawKnowledgeChunk(
                chunk_id=file_node_id,
                chunk_type=ChunkType.TEXT,
                content=content,
                source=relative_path,
                metadata={"file_path": file_path}
            )
            
            return [chunk]
            
        except Exception as e:
            logger.error(f"Failed to ingest {file_path}: {e}")
            return []

# Singleton implementation for global access
_global_ingestor = None

def get_knowledge_ingestor() -> KnowledgeIngestor:
    global _global_ingestor
    if _global_ingestor is None:
        _global_ingestor = KnowledgeIngestor()
    return _global_ingestor

if __name__ == "__main__":
    ingestor = get_knowledge_ingestor()
    ingestor.scan_and_ingest()
