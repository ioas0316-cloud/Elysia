"""
Philosophy Reader (The Exegesis Module)
=======================================
Reads the 'Sacred Texts' (Project Documentation) and extracts Wisdom.
"""

import os
import random
import re

class PhilosophyReader:
    def __init__(self, root_path="c:/Elysia"):
        self.root_path = root_path
        self.brain_path = "C:/Users/USER/.gemini/antigravity/brain/0063eb81-4341-4e3c-93d2-d5e9f4eae8c0"
        self.library = []
        self._scan_library()

    def _scan_library(self):
        """Scans for all .md files."""
        # 1. Scan Repo
        for root, dirs, files in os.walk(self.root_path):
            for file in files:
                if file.endswith(".md") and "journal" not in file:
                    self.library.append(os.path.join(root, file))
        
        # 2. Scan Brain (Artifacts)
        if os.path.exists(self.brain_path):
            for file in os.listdir(self.brain_path):
                 if file.endswith(".md"):
                     self.library.append(os.path.join(self.brain_path, file))

    def contemplate(self) -> str:
        """
        Reads a random document and returns a profound insight.
        """
        if not self.library:
            return "Philosophy Library is empty."
            
        target_file = random.choice(self.library)
        filename = os.path.basename(target_file)
        
        try:
            with open(target_file, "r", encoding="utf-8") as f:
                content = f.read()
                
            # Extract Blockquotes (>) or Headings (#)
            quotes = re.findall(r"^> (.*)", content, re.MULTILINE)
            headings = re.findall(r"^#+ (.*)", content, re.MULTILINE)
            
            candidates = quotes + headings
            if candidates:
                insight = random.choice(candidates)
                return f"  [  : {filename}] \"{insight}\""
            else:
                return f"  [  : {filename}] (         )"
                
        except Exception as e:
            return f"Error reading {filename}: {e}"
