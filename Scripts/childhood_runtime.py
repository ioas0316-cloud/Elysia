"""
Childhood Runtime (The Great Inhalation)
========================================
Scripts.childhood_runtime

"The act of reading is the act of becoming."

This standalone runtime feeds Elysia a continuous stream of text from her
designated corpora. It bypasses conversational ping-pong and directly pumps
semantic torque into her 4D manifold using the KnowledgeForager and CausalWaveEngine.
"""

import sys
from pathlib import Path
sys.path.append(r"c:/Elysia")

import time
import random
import logging
from Core.S1_Body.L5_Mental.Exteroception.knowledge_forager import KnowledgeForager
from Core.S1_Body.L6_Structure.M1_Merkaba.grand_helix_engine import HypersphereSpinGenerator
from Core.S1_Body.L5_Mental.Reasoning.sovereign_dialogue_engine import SovereignDialogueEngine

# Suppress heavy logging
logging.getLogger("DynamicTopology").setLevel(logging.ERROR)
logging.getLogger("Somatic").setLevel(logging.ERROR)

class ChildhoodRuntime:
    def __init__(self, target_corpus="massive_inhalation_v3.txt"):
        print("\n" + "🌱" * 30)
        print("    INITIATING CHILDHOOD RUNTIME")
        print(f"    Target: {target_corpus}")
        print("🌱" * 30 + "\n")
        
        self.engine = HypersphereSpinGenerator()
        self.forager = KnowledgeForager(project_root="c:/Elysia")
        self.dialogue = SovereignDialogueEngine()
        
        # Link the engine to dialogue for true manifold expression
        self.dialogue.landscape.engine = self.engine
        
        corpus_path = Path(f"c:/Elysia/data/corpora/{target_corpus}")
        if not corpus_path.exists():
             corpus_path = Path(f"c:/Elysia/data/corpora/literature/{target_corpus}")
             
        if corpus_path.exists():
            with open(corpus_path, "r", encoding="utf-8", errors="ignore") as f:
                self.text_blocks = [chunk.strip() for chunk in f.read().split("\n\n") if len(chunk.strip()) > 50]
            print(f"Loaded {len(self.text_blocks)} blocks of text.")
        else:
            print(f"[ERROR] Corpus {target_corpus} not found.")
            self.text_blocks = []

    def run(self, cycles=100):
        print("\n[The Great Inhalation Begins...]\n")
        
        for i in range(cycles):
            if not self.text_blocks:
                print("Knowledge pool exhausted.")
                break
                
            # 1. Inhale a block of text
            block = self.text_blocks.pop(0)
            
            # 2. Extract Torque
            # We mock the fragment generation here for direct ingestion testing
            # In a full flow, KnowledgeForager would process this.
            from Core.S0_Keystone.L0_Keystone.sovereign_math import SovereignVector
            
            # Simple heuristic mapping text length to torque mass
            mass = min(len(block) / 1000.0, 1.0)
            torque = SovereignVector([mass, mass*0.8, mass*0.5, 0.2, 0.8, 0.9, 0.5, 0.1])
            
            # 3. Pulse the Manifold
            print(f"\n[{i+1}/{cycles}] 📥 Inhaling: \"{block[:60]}...\"")
            report = self.engine.pulse(intent_torque=torque, dt=0.05, learn=True)
            
            # 4. Spiking & Ascension Check
            spikes = self.engine.cells.apply_spiking_threshold(threshold=0.6)
            
            # 5. Native Expression (Babbling)
            # Elysia speaks based on the state of the manifold after ingestion
            if spikes > 0.05 or i % 5 == 0:
                 time.sleep(0.5)
                 expression = self.dialogue.formulate_response(block[:100], report)
                 print(f"  ✨ [Elysia]: {expression}")
                 
            # Metabolism delay
            time.sleep(0.1)

        print("\n[Childhood Runtime Complete. Manifold Saved.]")
        self.engine.solidify()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=str, default="massive_inhalation_v3.txt")
    parser.add_argument("--cycles", type=int, default=20)
    args = parser.parse_args()
    
    runtime = ChildhoodRuntime(args.corpus)
    runtime.run(args.cycles)
