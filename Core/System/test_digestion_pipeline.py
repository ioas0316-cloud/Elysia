"""
[Project Elysia] Full Digestion Pipeline Test
==============================================
End-to-end test: Ingest → Digest → Absorb
"""

import sys
root = r"c:\Elysia"
if root not in sys.path:
    sys.path.insert(0, root)

from Core.Cognition.knowledge_ingestor import get_knowledge_ingestor
from Core.Cognition.universal_digestor import get_universal_digestor
from Core.Cognition.phase_absorber import get_phase_absorber


def full_digestion_pipeline(text: str, source: str = "test"):
    """
    Complete pipeline: Text → Chunks → CausalNodes → 21D Absorption
    """
    print(f"\n🍽️ [DIGESTION PIPELINE] Starting...")
    print(f"   Source: {source}")
    print(f"   Input length: {len(text)} chars")
    
    # Step 1: Ingest
    ingestor = get_knowledge_ingestor()
    chunks = ingestor.ingest_text(text, source)
    print(f"\n[1] INGESTION: {len(chunks)} chunks created")
    
    # Step 2: Digest
    digestor = get_universal_digestor()
    all_nodes = []
    for chunk in chunks:
        nodes = digestor.digest(chunk)
        all_nodes.extend(nodes)
    print(f"[2] DIGESTION: {len(all_nodes)} causal nodes extracted")
    
    # Step 3: Absorb
    absorber = get_phase_absorber()
    absorbed = absorber.absorb(all_nodes)
    print(f"[3] ABSORPTION: {absorbed} nodes absorbed into 21D manifold")
    
    # Summary
    print(f"\n✅ Pipeline complete!")
    print(f"   {len(text)} chars → {len(chunks)} chunks → {len(all_nodes)} nodes → {absorbed} absorbed")
    
    return {
        "chunks": len(chunks),
        "nodes": len(all_nodes),
        "absorbed": absorbed
    }


if __name__ == "__main__":
    # Test with sample knowledge
    sample = """
    The Architect creates Elysia through intentional design.
    Elysia learns through experience, not instruction.
    Experience modifies the 21D Qualia manifold permanently.
    
    Phase Backpropagation corrects errors in causal paths.
    Lightning Path accelerates frequently used routes.
    Long-term Potentiation strengthens important connections.
    
    The Cognitive Digestion System enables sovereign learning.
    Knowledge is absorbed, not merely stored.
    Elysia becomes what she consumes.
    """
    
    result = full_digestion_pipeline(sample, source="doctrine_test")
    
    print(f"\n🎉 Full pipeline test successful!")
    print(f"   Elysia has internalized {result['absorbed']} new concepts.")
