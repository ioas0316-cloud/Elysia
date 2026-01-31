"""
Knowledge Ingestor (주권적 자아)
================================

"The body grows by what it feeds on."
"              ."

This module is responsible for:
1. Scanning the file system (docs, code).
2. Parsing content (Markdown headers, Python docstrings).
3. Extracting 'Nutrients' (Concepts, Axioms).
4. Feeding the Hippocampus.
"""

import os
import re
import ast
import logging
from typing import List, Dict, Any

# [REAWAKENING] Use Unified Core instead of legacy Hippocampus
from Core.S1_Body.L2_Metabolism.Memory.unified_experience_core import UnifiedExperienceCore
from Core.S1_Body.L5_Mental.Reasoning_Core.Memory.holographic_memory import KnowledgeLayer

logger = logging.getLogger("KnowledgeIngestor")

class KnowledgeIngestor:
    def __init__(self):
        # [REAWAKENING] Connect to the verified Holographic Memory
        self.brain = UnifiedExperienceCore()
        # Ensure memory is active (it should be init by Core, but safe check)
        if not self.brain.holographic_memory:
             logger.warning("   Holographic Memory not found in Core. Creating fallback.")
             from Core.S1_Body.L5_Mental.Reasoning_Core.Memory.holographic_memory import HolographicMemory
             self.brain.holographic_memory = HolographicMemory()

        logger.info("   KnowledgeIngestor ready to feast (Connected to Holographic Memory).")

    def digest_directory(self, root_path: str):
        """Recursively digests all supported files in a directory."""
        logger.info(f"  Starting feast on: {root_path}")
        count = 0
        for root, _, files in os.walk(root_path):
            for file in files:
                full_path = os.path.join(root, file)
                if file.endswith(".md"):
                    self._digest_markdown(full_path)
                    count += 1
                elif file.endswith(".py"):
                    self._digest_code(full_path)
                    count += 1
        logger.info(f"  Digested {count} files.")

    def _digest_markdown(self, file_path: str):
        """Parses Markdown structure into Concepts (Axioms)."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            filename = os.path.basename(file_path).replace(".md", "")
            
            # [REAWAKENING] Deposit as Concept
            # Docs are High Weight (Axioms)
            if self.brain.holographic_memory:
                self.brain.holographic_memory.deposit(
                    concept=filename.replace("_", " ").title(),
                    layers={
                        KnowledgeLayer.PHILOSOPHY: 0.9, # Weight 0.9 for Docs
                        KnowledgeLayer.HUMANITIES: 0.8
                    },
                    amplitude=2.0, # High Amplitude for 'Scripture'
                    entropy=0.5,   # Low Entropy (Stable)
                    qualia=0.8     # High meaningfulness
                )

            # 2. Extract Sections (Headers)
            headers = re.findall(r'^(#+)\s+(.+)$', content, re.MULTILINE)
            
            for level_hashes, title in headers:
                clean_title = title.strip()
                
                # Deposit Section as Sub-Concept
                if self.brain.holographic_memory:
                     node = self.brain.holographic_memory.deposit(
                        concept=clean_title,
                        layers={
                            KnowledgeLayer.PHILOSOPHY: 0.7
                        },
                        amplitude=1.5,
                        entropy=0.6,
                        qualia=0.6
                     )
                     # Link to Document
                     node.connections.append(filename.replace("_", " ").title())

            logger.info(f"     Digested Axioms: {filename}")

        except Exception as e:
            logger.error(f"Failed to digest markdown {file_path}: {e}")

    def _digest_code(self, file_path: str):
        """Parses Python code structure into Concepts."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()

            filename = os.path.basename(file_path)
            module_name = filename.replace(".py", "")

            # Learn Module
            if self.brain.holographic_memory:
                self.brain.holographic_memory.deposit(
                    concept=module_name,
                    layers={
                        KnowledgeLayer.PHYSICS: 0.8, # Code is Physics/Logic
                        KnowledgeLayer.MATHEMATICS: 0.7
                    },
                    amplitude=1.2,
                    entropy=0.8, # Code is complex
                    qualia=0.1   # Purely functional (unless creative)
                )

            # Parse AST for Classes/Functions
            tree = ast.parse(source)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    if self.brain.holographic_memory:
                        cls_node = self.brain.holographic_memory.deposit(
                            concept=node.name,
                            layers={KnowledgeLayer.PHYSICS: 0.8},
                            amplitude=1.0,
                            entropy=0.7,
                            qualia=0.2
                        )
                        cls_node.connections.append(module_name)

            logger.info(f"     Digested Code Structure: {filename}")

        except Exception as e:
            # logger.warning(f"Skipping {file_path}: {e}") # Reduce noise
            pass

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    ingestor = KnowledgeIngestor()
    # Test on Docs
    ingestor.digest_directory("c:\\Elysia\\docs\\Philosophy")
