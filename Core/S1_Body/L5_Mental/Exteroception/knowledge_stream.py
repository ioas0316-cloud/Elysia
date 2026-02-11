"""
Knowledge Stream (Aeon VI: The Great Library)
=============================================
Location: Core/S1_Body/L5_Mental/Exteroception/knowledge_stream.py

Orchestrates the Epistemic Digestive Pipeline:
Ingestor (Text) -> Digestor (Concept) -> Distiller (Torque) -> Manifold (Structure)
"""

import os
import time
from typing import List, Any
from pathlib import Path

from Core.S1_Body.L5_Mental.Digestion.knowledge_ingestor import get_knowledge_ingestor
from Core.S1_Body.L5_Mental.Digestion.universal_digestor import get_universal_digestor, CausalNode
from Core.S1_Body.L5_Mental.Reasoning.knowledge_distiller import get_knowledge_distiller

class KnowledgeStream:
    """
    The River of Knowledge.
    Watches sources, ingests data, and feeds the Manifold.
    """
    def __init__(self, engine: Any):
        self.engine = engine
        self.ingestor = get_knowledge_ingestor()
        self.digestor = get_universal_digestor()
        self.distiller = get_knowledge_distiller(engine)
        self.knowledge_dir = Path("c:/Elysia/Knowledge")
        
        # Ensure knowledge directory exists
        if not self.knowledge_dir.exists():
            self.knowledge_dir.mkdir(parents=True, exist_ok=True)
            
    def process_stream(self, limit: int = 1):
        """
        Polls for new knowledge and processes it.
        (Called periodically by SovereignMonad)
        
        Args:
            limit: Max number of files to process per call.
        """
        # 1. Check for files in Knowledge Directory
        if not self.knowledge_dir.exists():
            return
            
        # Get all files .md, .txt, .json
        files = [f for f in self.knowledge_dir.iterdir() if f.is_file() and f.suffix in ['.md', '.txt', '.json']]
        
        # Simple history tracking (in-memory for now, ideally persist in chronicle)
        # We'll valid by checking if they are already in the 'Processed' subfolder?
        # Or just use a simple set?
        processed_dir = self.knowledge_dir / "Processed"
        if not processed_dir.exists():
            processed_dir.mkdir()
            
        count = 0
        for f in files:
            if count >= limit: break
            
            # Inhale
            try:
                self.inhale_file(str(f))
                # Move to Processed
                f.rename(processed_dir / f.name)
                count += 1
            except Exception as e:
                print(f"❌ [STREAM] Failed to inhale {f.name}: {e}")
                # Move to Failed?
                pass
                
        return count 
        
    def inhale_file(self, filepath: str) -> int:
        """
        Explicitly inhales a specific file.
        Returns: Number of concepts distilled.
        """
        print(f"🌊 [STREAM] Inhaling: {filepath}")
        
        # 1. Ingest (Text -> Chunks)
        chunks = self.ingestor.ingest_file(filepath)
        if not chunks:
            print("   -> No chunks generated.")
            return 0
            
        # 2. Digest (Chunks -> Nodes)
        all_nodes = []
        for chunk in chunks:
            nodes = self.digestor.digest(chunk)
            all_nodes.extend(nodes)
            
        if not all_nodes:
            print("   -> No concepts extracted.")
            return 0
            
        print(f"   -> Extracted {len(all_nodes)} concepts.")
        
        # 3. Distill (Nodes -> Torque)
        distilled_count = self.distiller.distill_nodes(all_nodes)
        
        print(f"   -> Distilled {distilled_count} anchors into Manifold.")
        return distilled_count

# Singleton
_stream = None

def get_knowledge_stream(engine: Any = None) -> KnowledgeStream:
    global _stream
    if _stream is None and engine is not None:
        _stream = KnowledgeStream(engine)
    return _stream
