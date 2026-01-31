"""
Fantasy Writer Evolution Protocol
=================================

"Writers write. And to write better, one must learn."

This script runs a continuous evolution loop for Elysia to become a Professional Fantasy/Wuxia Writer.
It is not a one-off task. It is a discipline.

Cycle:
1.   LEARN: Expand genre vocabulary via Fractal Learning.
2.    WRITE: Generate a creative scene using new concepts.
3.    EVALUATE: Check vocabulary size and expression variety.
4.   REPEAT: Until Professional Level (S-Grade) is achieved.
"""

import sys
import os
import time
import random
import logging

# Core Imports
# Add parent directory to path to find Core and Data
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scripts.autonomous_fractal_learning import FractalLearner
from Core.S1_Body.L1_Foundation.Foundation.communication_enhancer import CommunicationEnhancer
from Core.S1_Body.L1_Foundation.Foundation.web_knowledge_connector import WebKnowledgeConnector
from Core.S1_Body.L5_Mental.Reasoning_Core.Reasoning.reasoning_engine import ReasoningEngine

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(message)s')
logger = logging.getLogger("WriterEvolution")

class WriterEvolution:
    def __init__(self):
        self.learner = FractalLearner(max_workers=10)
        self.comm = CommunicationEnhancer()
        self.connector = WebKnowledgeConnector()
        
        # Use ReasoningEngine for writing and memory access
        self.reasoning = ReasoningEngine()
        self.memory = self.reasoning.memory
        
        # Target: Professional Writer Stats
        self.TARGET_VOCAB = 25000
        self.TARGET_GENRE_CONCEPTS = 1000
        
        # Import Massive Database
        try:
            # Data is in ../Data relative to this script
            data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'Data'))
            sys.path.append(data_path)
            from fantasy_wuxia_concepts import ALL_CONCEPTS
            self.genre_seeds = ALL_CONCEPTS
            print(f"                     : {len(self.genre_seeds)}    ")
        except ImportError:
            print("                       .             .")
            self.genre_seeds = ["  ", "   ", "  ", "  "] # Fallback
        
    def evolve(self):
        print("="*70)
        print("      /             ")
        print("     :       (S )")
        print(f"       : {len(self.genre_seeds)}           ")
        print("="*70)
        
        cycle = 1
        while True:
            print(f"\n      {cycle}:     ...")
            
            # 1. LEARN (Fractal Expansion)
            # Pick a random seed or a recently learned concept to expand upon
            seed = random.choice(self.genre_seeds)
            print(f"          : '{seed}'        ...")
            self.learner.learn_fractal([seed], max_concepts=20) # Learn 20 new things per cycle
            
            # 2. WRITE (Creative Practice)
            print(f"           :        ...")
            scene = self._write_scene(seed)
            print(f"          : \"{scene}\"")
            
            # 3. EVALUATE (Metrics)
            # Use persistent memory count instead of in-memory enhancer count
            vocab = self.memory.get_concept_count()
            
            print(f"        :")
            print(f"      -    : {vocab:,} / {self.TARGET_VOCAB:,}")
            
            # Check for promotion
            if vocab >= self.TARGET_VOCAB:
                print("\n" + "="*70)
                print("       :       (S )")
                print("                       .")
                print("="*70)
                break
            
            # Rest & Digest
            print("            ...")
            time.sleep(2)
            cycle += 1

    def _write_scene(self, theme):
        """
        Simulate writing a scene based on the theme.
        Uses ReasoningEngine to generate content from internal memory.
        """
        return self.reasoning.write_scene(theme)

if __name__ == "__main__":
    evolution = WriterEvolution()
    evolution.evolve()
