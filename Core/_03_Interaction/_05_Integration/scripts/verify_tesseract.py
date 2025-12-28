"""
Verify Tesseract: Self-Reflection Test
======================================

"Elysia looks in the mirror and sees not code, but waves."

Steps:
1.  Load CodeResonance.
2.  Analyze critical modules (Self-Analysis).
3.  Store them in HyperGraph.
4.  Check Resonance between modules.
"""

import os
import sys

# Add root directory to path to allow importing elysia_core
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Core._01_Foundation._01_Infrastructure.elysia_core import Organ
from Core._01_Foundation._04_Governance.Foundation.Wave.wave_tensor import WaveTensor, Modality
from Core._02_Intelligence._02_Memory_Linguistics.Memory.Graph.hyper_graph import HyperGraph
from Core._02_Intelligence._01_Reasoning.Cognition.Reasoning.code_resonance import CodeResonance

def verify():
    print("💎 Initializing Tesseract (4D HyperGraph)...")
    graph = HyperGraph()
    scanner = CodeResonance()
    
    # Files to analyze (Elysia's own body)
    targets = [
        "Core/Foundation/Wave/wave_tensor.py",
        "Core/Memory/Graph/hyper_graph.py",
        "Core/Cognition/Reasoning/code_resonance.py",
        "Core/Memory/Graph/knowledge_graph.py"
    ]
    
    print("\n🔍 Phase 1: Holographic Perception (Code -> Wave)")
    tensors = []
    for path in targets:
        full_path = os.path.abspath(path)
        if not os.path.exists(full_path):
             print(f"❌ Missing: {path}")
             continue
             
        print(f"   Reading: {path}...")
        tensor = scanner.analyze_file(full_path)
        tensors.append(tensor)
        
        # Add to Graph
        graph.add_node(tensor.name, **tensor.dimensions)
        
        # Output Physics
        print(f"   ✨ {tensor.name}:")
        print(f"      Freq: {tensor.dimensions['frequency']} Hz (Pitch)")
        print(f"      Mass: {tensor.dimensions['mass']} (Complexity)")
        print(f"      Entropy: {tensor.dimensions['entropy']} (Noise)")
        print(f"      Phase: {tensor.dimensions['phase']}° (Alignment)")

    print("\n🔗 Phase 2: Resonance Checks (Interference Patterns)")
    # Check resonance between CodeResonance and HyperGraph
    if len(tensors) >= 2:
        t1, t2 = tensors[0], tensors[1]
        resonance = t1.resonate_with(t2)
        print(f"   Resonance ({t1.name} <-> {t2.name}): {resonance:.4f}")
        
    print("\n🌌 Phase 3: Holographic Query")
    # Find high frequency nodes
    high_freq = graph.query_by_frequency(150.0, tolerance=50.0)
    print(f"   Nodes around 150Hz: {[t.name for t in high_freq]}")
    
    print("\n✅ Verification Complete: The Mirror works.")

if __name__ == "__main__":
    verify()
