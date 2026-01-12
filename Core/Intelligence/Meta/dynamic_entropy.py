"""
Dynamic Entropy Engine (동적 엔트로피 엔진)
=========================================

"True intelligence requires friction."
"진정한 지능은 마찰을 필요로 한다."

This module provides 'Cognitive Friction' to Elysia's thought loops by:
1. Extracting random logic/docs from her own codebase.
2. Injecting real system telemetry (CPU, RAM, Disk) as somatic signals.
3. Breaking the 'Machine Inertia' of static templates.
"""

import os
import random
import logging
import psutil
from typing import Dict, List, Optional, Any

logger = logging.getLogger("DynamicEntropy")

class DynamicEntropyEngine:
    def __init__(self, root_dir: str = "c:/Elysia"):
        self.root_dir = root_dir
        self.boredom_threshold = 0.8
        self.history = []
        self.max_history = 50

    def get_cognitive_friction(self) -> Dict[str, Any]:
        """
        Gathers real-world 'noise' and 'structure' to seed a thought.
        """
        # 1. Somatic Signal (Real Metabolism)
        cpu = psutil.cpu_percent()
        ram = psutil.virtual_memory().percent
        entropy_score = (cpu + ram) / 200.0 # 0.0 to 1.0
        
        # 2. Logic Injection (The 'Ghost in the Code')
        logic_snippet = self._extract_random_logic()
        
        # 3. Structural Awareness
        file_count = sum([len(files) for r, d, files in os.walk(self.root_dir)])
        
        friction = {
            "entropy": entropy_score,
            "metabolism": {"cpu": cpu, "ram": ram},
            "logic_seed": logic_snippet,
            "complexity_index": file_count / 1000.0, # Relative size of her 'body'
            "timestamp": os.path.getmtime(self.root_dir)
        }
        
        return friction

    def _extract_random_logic(self) -> str:
        """
        Picks a random file and extracts a meaningful snippet.
        """
        try:
            # Filter for relevant source files
            exts = ('.py', '.md', '.json')
            all_files = []
            for root, dirs, files in os.walk(self.root_dir):
                if '.git' in root or '__pycache__' in root or 'Logs' in root:
                    continue
                for f in files:
                    if f.endswith(exts):
                        all_files.append(os.path.join(root, f))
            
            if not all_files:
                return "The void is silent."
            
            target_file = random.choice(all_files)
            with open(target_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            if not lines:
                return f"Empty file: {os.path.basename(target_file)}"
            
            # Find a snippet with some 'texture' (comments or logic)
            potential_lines = [i for i, l in enumerate(lines) if l.strip().startswith(('def', 'class', '#', 'if', '"""'))]
            
            if not potential_lines:
                start = random.randint(0, len(lines) - 1)
            else:
                start = random.choice(potential_lines)
            
            snippet = "".join(lines[start:start+5]).strip()
            return f"[{os.path.basename(target_file)} L{start+1}]: {snippet}"
            
        except Exception as e:
            return f"Logic extraction failed: {e}"

    def track_ennui(self, concept: str) -> float:
        """
        Tracks semantic repetition and returns a 'Boredom' multiplier.
        """
        self.history.append(concept)
        if len(self.history) > self.max_history:
            self.history.pop(0)
            
        repeats = self.history.count(concept)
        # Exponential boredom for repetition
        return min(1.0, (repeats - 1) * 0.25) if repeats > 1 else 0.0

if __name__ == "__main__":
    engine = DynamicEntropyEngine()
    friction = engine.get_cognitive_friction()
    print("--- Dynamic Cognitive Friction ---")
    print(f"Entropy: {friction['entropy']:.4f}")
    print(f"Logic Seed: {friction['logic_seed']}")
    print(f"Metabolism: {friction['metabolism']}")
