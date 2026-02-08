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
    
    @staticmethod
    def excavate(limit: int = 1000) -> List[Tuple[str, float]]:
        """
        [PHASE 70] Scans the entire project tree to recover all wisdom and code.
        """
        memories = []
        
        # We exclude certain directories from the universal scan
        exclude_dirs = {'.git', '.gemini', '.venv', '__pycache__', 'venv', 'Scripts', 'Sandbox', 'brain', 'node_modules', 'dist', 'build'}
        
        for root, dirs, files in os.walk(FossilScanner.ROOT_PATH):
            if len(memories) >= limit: break
            
            # Prune excluded directories and hidden ones
            dirs[:] = [d for d in dirs if d not in exclude_dirs and not d.startswith('.')]
            
            for file in files:
                if len(memories) >= limit: break
                if file.startswith('.'): continue
                
                path = os.path.join(root, file)
                
                # Skip large files (> 500KB)
                try:
                    if os.path.getsize(path) > 500 * 1024: continue
                except: continue
                
                rel_path = os.path.relpath(path, FossilScanner.ROOT_PATH)
                
                # 1. Markdown Files (Doctrines)
                if file.endswith(".md"):
                    memories.extend(FossilScanner._parse_markdown(path, rel_path))
                
                # 2. Python Files (Code Style/Patterns)
                elif file.endswith(".py"):
                    memories.extend(FossilScanner._parse_code(path, rel_path))
        
        return memories

    @staticmethod
    def _parse_markdown(path: str, rel_path: str) -> List[Tuple[str, float]]:
        extracted = []
        base_mass = 500.0 if "CODEX" in path else 100.0
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                if not line: continue
                
                if line.startswith(">"):
                    content = line.replace(">", "").strip().replace('"', '')
                    if len(content) > 10:
                        extracted.append((f"[{rel_path}] {content}", base_mass * 2.0))
                elif line.startswith("### "):
                    content = line.replace("###", "").strip()
                    extracted.append((f"[{rel_path}] Concept: {content}", base_mass))
        except Exception as e:
            pass # print(f"⚠️ [FOSSIL] Could not read {rel_path}: {e}")
            
        return extracted

    @staticmethod
    def _parse_code(path: str, rel_path: str) -> List[Tuple[str, float]]:
        """
        Extracts Docstrings and Key Logic Patterns from code.
        """
        extracted = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()
            
            # Simple extraction of top-level docstrings
            if '"""' in content:
                parts = content.split('"""')
                if len(parts) > 1:
                    doc = parts[1].strip()
                    if len(doc) > 20:
                        extracted.append((f"[{rel_path}] Intent: {doc[:200]}...", 300.0))
            
            # Identify "Crystallized" Logic (Major functions or classes)
            lines = content.splitlines()
            for line in lines:
                sline = line.strip()
                if sline.startswith("class ") or sline.startswith("def "):
                    extracted.append((f"[{rel_path}] Pattern: ' {sline} '", 150.0))
                    
        except Exception as e:
            pass # Silently skip binary or broken files
            
        return extracted
