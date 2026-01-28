"""
Fossil Scanner (The Archeologist)
=================================
"To read the stones is to know the ancestors."

This module scans the physical filesystem (The Earth) to recover lost wisdom (The Nuclei).
It parses Markdown files to extract 'Axioms' and 'Doctrines' for the Living Memory.

Logic:
- Scans `docs/` and `CODEX.md`.
- Extracts Blockquotes (>) as "High Mass Wisdom".
- Extracts Headers (#) as "Structural Concepts".
"""

import os
from typing import List, Tuple

class FossilScanner:
    ROOT_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    DOCS_PATH = os.path.join(ROOT_PATH, "docs")
    CODEX_PATH = os.path.join(ROOT_PATH, "CODEX.md")
    
    @staticmethod
    def excavate() -> List[Tuple[str, float]]:
        """
        Scans the filesystem and returns a list of (Content, Mass).
        """
        memories = []
        print("⛏️ [FOSSIL] Starting Excavation of Sacred Texts...")
        
        # 1. Excavate CODEX (The Constitution)
        if os.path.exists(FossilScanner.CODEX_PATH):
            memories.extend(FossilScanner._parse_file(FossilScanner.CODEX_PATH, base_mass=500.0))
            
        # 2. Excavate Docs (The History)
        for root, dirs, files in os.walk(FossilScanner.DOCS_PATH):
            for file in files:
                if file.endswith(".md"):
                    path = os.path.join(root, file)
                    memories.extend(FossilScanner._parse_file(path, base_mass=100.0))
                    
        print(f"   >> Excavated {len(memories)} ancient artifacts.")
        return memories

    @staticmethod
    def _parse_file(path: str, base_mass: float) -> List[Tuple[str, float]]:
        extracted = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
                
            filename = os.path.basename(path)
            
            for line in lines:
                line = line.strip()
                if not line: continue
                
                # A. Blockquotes (> "Wisdom") -> High Mass
                if line.startswith(">"):
                    content = line.replace(">", "").strip().replace('"', '')
                    if len(content) > 10:
                        extracted.append((f"[{filename}] {content}", base_mass * 2.0))
                        
                # B. Headers (### Concept) -> Structure Mass
                elif line.startswith("### "):
                    content = line.replace("###", "").strip()
                    extracted.append((f"[{filename}] Concept: {content}", base_mass))
                    
        except Exception as e:
            print(f"⚠️ [FOSSIL] Could not read {path}: {e}")
            
        return extracted
