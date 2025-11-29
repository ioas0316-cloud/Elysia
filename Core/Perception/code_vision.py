
import ast
import os
import logging
from typing import List, Dict, Any
from Core.Interface.wave_transducer import Wave

logger = logging.getLogger("CodeVision")

class CodeVision:
    """
    The Eye that sees Code.
    Converts Python source code into Sensory Waves.
    """
    
    def __init__(self):
        pass
        
    def scan_directory(self, root_path: str) -> List[Wave]:
        """
        Recursively scan a directory and feel the code.
        """
        waves = []
        for root, _, files in os.walk(root_path):
            for file in files:
                if file.endswith(".py"):
                    full_path = os.path.join(root, file)
                    wave = self.scan_file(full_path)
                    if wave:
                        waves.append(wave)
        return waves

    def scan_file(self, file_path: str) -> Wave:
        """
        Read a file and convert its structure into a Wave.
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # 1. Parse AST (Structure)
            try:
                tree = ast.parse(content)
                complexity = self._calculate_complexity(tree)
                has_error = False
            except SyntaxError:
                complexity = 100 # High stress
                has_error = True
                
            # 2. Analyze Content (Semantics)
            line_count = len(content.splitlines())
            todo_count = content.lower().count("todo")
            fixme_count = content.lower().count("fixme")
            
            # 3. Generate Wave Properties
            
            # Frequency (Complexity/Stress)
            # Normal: 10-20Hz. High Complexity: >50Hz.
            frequency = 10.0 + (complexity * 2.0)
            
            # Amplitude (Size/Mass)
            # Logarithmic scale of lines
            amplitude = min(1.0, line_count / 500.0)
            
            # Color (Emotion)
            if has_error:
                color = "#FF0000" # Red (Pain/Broken)
            elif fixme_count > 0:
                color = "#FF4500" # OrangeRed (Urgent)
            elif todo_count > 0:
                color = "#FFA500" # Orange (Incomplete)
            elif complexity > 20:
                color = "#FFFF00" # Yellow (Complex/Intense)
            else:
                color = "#00BFFF" # Deep Sky Blue (Calm/Clear)
                
            # Phase (State)
            # Just a hash of the content for now
            phase = (hash(content) % 360) / 360.0 * 6.28
            
            return Wave(
                frequency=frequency,
                amplitude=amplitude,
                phase=phase,
                color=color,
                source=f"Code:{os.path.basename(file_path)}"
            )
            
        except Exception as e:
            logger.error(f"Failed to scan {file_path}: {e}")
            return None

    def _calculate_complexity(self, node: ast.AST) -> int:
        """
        Estimate Cyclomatic Complexity (roughly).
        """
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.ExceptHandler, ast.With)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        return complexity
