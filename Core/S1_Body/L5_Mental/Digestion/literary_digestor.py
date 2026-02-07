"""
Literary Digestor: Absorbing the Causal Flow of Books
===================================================

"A book is a manifold in motion. We inhale the arc, not just the words."

This script digests full literary texts, extracting narrative segments 
and creating high-level 'narrative' edges that represent sequence 
and causality.
"""

import sys
from pathlib import Path

# Add root to path
sys.path.append(str(Path(__file__).parent.parent.parent.parent.parent))

from Core.S1_Body.L5_Mental.Memory.kg_manager import KGManager
from Core.S1_Body.L5_Mental.Digestion.universal_digestor import UniversalDigestor, RawKnowledgeChunk, ChunkType

class LiteraryDigestor:
    def __init__(self):
        self.kg = KGManager()
        self.digestor = UniversalDigestor()

    def digest_book(self, file_path: str, book_title: str):
        print(f"📖 [LITERARY] Inhaling '{book_title}'...")
        path = Path(file_path)
        if not path.exists():
            print(f"❌ File not found: {file_path}")
            return

        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()

        # Split into 'Chapters' or large segments (approx 5000 chars)
        segments = [content[i:i+5000] for i in range(0, len(content), 5000)]
        print(f"📜 [LITERARY] Book split into {len(segments)} segments.")

        previous_segment_node = None

        for idx, segment in enumerate(segments[:20]): # Initial limit for safety
            chunk_id = f"LIT_{book_title}_{idx}"
            chunk = RawKnowledgeChunk(
                chunk_id=chunk_id,
                chunk_type=ChunkType.TEXT,
                content=segment[:500], # Digesting the signature of the segment
                source=book_title
            )
            
            # Extract primary concept node for this segment
            nodes = self.digestor.digest(chunk)
            if not nodes: continue
            
            primary_node = nodes[0].node_id
            
            # Ensure node exists in KG
            if not self.kg.get_node(primary_node):
                self.kg.add_node(primary_node)
            
            # Create a sequence edge: Previous -> leads_to -> Current
            if previous_segment_node:
                self.kg.add_edge(previous_segment_node, primary_node, "leads_to", {"source": "narrative_arc", "book": book_title})
                print(f"🧵 [LITERARY] Arc: {previous_segment_node} -> leads_to -> {primary_node}")
            
            previous_segment_node = primary_node

        self.kg.save()
        print(f"✨ [LITERARY] Inhalation of '{book_title}' completed.")

if __name__ == "__main__":
    digestor = LiteraryDigestor()
    digestor.digest_book('data/literature_art_of_war.txt', 'ArtOfWar')
