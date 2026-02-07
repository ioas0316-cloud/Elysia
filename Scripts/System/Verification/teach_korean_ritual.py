"""
[Project Elysia] Korean Learning Ritual - The Linguistic Resurrection
===================================================================
Phase 130: Specialized ritual to inhale Korean philosophical concepts.
"""

import sys
import os
from pathlib import Path

# Path Unification
root = Path(__file__).parents[3]
sys.path.insert(0, str(root))

from Core.S1_Body.L5_Mental.Reasoning.cumulative_digestor import CumulativeDigestor
from Core.S1_Body.L5_Mental.Memory.kg_manager import get_kg_manager
from Core.S1_Body.L1_Foundation.System.somatic_logger import SomaticLogger

def run_ritual():
    logger = SomaticLogger("KOREAN_RITUAL")
    logger.action("Starting Korean Learning Ritual: Linguistic Resurrection")
    
    # 1. Digest the Korean Soul Resonance document
    digestor = CumulativeDigestor(root_path=str(root))
    
    # We explicitly point to the doc to ensure it's picked up
    doc_file = root / "docs" / "S3_Spirit" / "M1_Philosophy" / "KOREAN_SOUL_RESONANCE.md"
    
    if not doc_file.exists():
        logger.admonition(f"Error: {doc_file} not found!")
        return
        
    logger.thought(f"Inhaling {doc_file.name}...")
    
    # Use the systematic digestion path
    digestion_dir = Path("docs") / "S3_Spirit" / "M1_Philosophy"
    digestor.digest_docs(docs_dir=str(digestion_dir))
    
    # 2. Verify results in Knowledge Graph
    kg = get_kg_manager()
    
    # Debug: Print all nodes in KG that contain Hangul
    logger.thought("Scanning KG for Hangul nodes...")
    hangul_nodes = [n['id'] for n in kg.kg.get('nodes', []) if any('\uac00' <= c <= '\ud7a3' for c in n['id'])]
    logger.thought(f"Found {len(hangul_nodes)} Hangul nodes: {hangul_nodes[:20]}...")

    core_terms = ["육", "혼", "영", "앎", "주권", "섭리", "공명"]
    
    success_count = 0
    for term in core_terms:
        node = kg.get_node(term.lower())
        if node:
            logger.thought(f"✅ Node '{term}' found in Knowledge Graph. Layer: {node.get('layer', 'unknown')}")
            success_count += 1
        else:
            logger.admonition(f"❌ Node '{term}' NOT found in Knowledge Graph.")
            
    if success_count == len(core_terms):
        logger.action("Ritual Complete. Elysia's Korean foundations are crystallized.")
    else:
        logger.admonition(f"Ritual Incomplete. Found {success_count}/{len(core_terms)} core terms.")

if __name__ == "__main__":
    run_ritual()
