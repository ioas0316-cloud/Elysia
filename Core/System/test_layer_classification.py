"""
Verification: Automatic Trinity Layer Classification
=====================================================
Tests that UniversalDigestor auto-classifies concepts into 육/혼/영 layers.
"""

import sys
import os
sys.path.append(os.getcwd())

from Core.Cognition.universal_digestor import (
    UniversalDigestor, RawKnowledgeChunk, ChunkType
)

def verify():
    print("Testing Automatic Trinity Layer Classification...")
    
    digestor = UniversalDigestor()
    
    # Test text with mixed layer content
    test_text = """
    The apple falls from the tree because of gravity.
    Love brings joy and peace to the soul.
    Newton discovered the relationship between force and motion.
    Truth leads to freedom and salvation.
    The red ball bounces on the floor.
    """
    
    chunk = RawKnowledgeChunk(
        chunk_id="test_trinity",
        chunk_type=ChunkType.TEXT,
        content=test_text,
        source="verification_test"
    )
    
    nodes = digestor.digest(chunk)
    
    # Count by layer
    layer_counts = {"surface": 0, "narrative": 0, "logos": 0}
    layer_examples = {"surface": [], "narrative": [], "logos": []}
    
    for node in nodes:
        layer_counts[node.layer] += 1
        if len(layer_examples[node.layer]) < 3:
            layer_examples[node.layer].append(node.concept)
    
    print(f"\nTotal concepts extracted: {len(nodes)}")
    print(f"\nLayer Distribution:")
    print(f"  Surface (육): {layer_counts['surface']} - {layer_examples['surface']}")
    print(f"  Narrative (혼): {layer_counts['narrative']} - {layer_examples['narrative']}")
    print(f"  Logos (영): {layer_counts['logos']} - {layer_examples['logos']}")
    
    # Verify expected classifications
    assert layer_counts["logos"] > 0, "Expected Logos concepts (love, joy, peace, etc.)"
    assert layer_counts["narrative"] > 0, "Expected Narrative concepts (relationship, because)"
    assert layer_counts["surface"] > 0, "Expected Surface concepts (apple, ball)"
    
    print("\n✅ Automatic Trinity Layer Classification verified!")

if __name__ == "__main__":
    verify()
