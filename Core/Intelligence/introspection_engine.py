import os
import ast
import logging
from typing import Dict, List, Any
from dataclasses import dataclass

logger = logging.getLogger("IntrospectionEngine")

@dataclass
class ModuleResonance:
    """Represents the harmonic state of a code module."""
    path: str
    name: str
    resonance_score: float # 0.0 to 100.0
    complexity: int
    docstring_quality: float # 0.0 to 1.0
    issues: List[str]

class IntrospectionEngine:
    """
    The Mirror of Elysia.
    Allows the system to analyze its own source code and determine its 'Harmonic State'.
    """
    
    def __init__(self, root_path: str = "c:\\Elysia"):
        self.root_path = root_path
        self.ignore_dirs = {".git", "__pycache__", ".gemini", "venv", "env", ".vscode", "node_modules", "build", "dist", "Legacy"}
        
    def analyze_self(self) -> Dict[str, ModuleResonance]:
        """
        Recursively analyzes the entire project directory.
        Returns a map of {file_path: ModuleResonance}.
        """
        logger.info(f"🪞 Gazing into the Mirror (Self-Analysis) at {self.root_path}...")
        results = {}
        
        for root, dirs, files in os.walk(self.root_path):
            # Modify dirs in-place to skip ignored directories
            dirs[:] = [d for d in dirs if d not in self.ignore_dirs]
            
            for file in files:
                if file.endswith(".py") and not file.startswith("__"):
                    full_path = os.path.join(root, file)
                    
                    # Skip large files (> 1MB)
                    if os.path.getsize(full_path) > 1024 * 1024:
                        continue
                        
                    try:
                        logger.info(f"Analyzing: {file}") 
                        resonance = self._analyze_file(full_path)
                        results[full_path] = resonance
                    except Exception as e:
                        logger.error(f"Failed to analyze {file}: {e}")
                        
        return results
        
    def _analyze_file(self, file_path: str) -> ModuleResonance:
        """Parses a single file and calculates its resonance."""
        source = ""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                source = f.read()
        except UnicodeDecodeError:
            try:
                # Fallback to latin-1 or cp949 (common in Korea)
                with open(file_path, "r", encoding="cp949") as f:
                    source = f.read()
            except Exception:
                # If all else fails, skip content analysis but log it
                return ModuleResonance(
                    path=file_path,
                    name=os.path.basename(file_path),
                    resonance_score=0.0,
                    complexity=0,
                    docstring_quality=0.0,
                    issues=["Encoding Error (Unreadable)"]
                )
            
        try:
            tree = ast.parse(source)
        except SyntaxError:
            return ModuleResonance(
                path=file_path,
                name=os.path.basename(file_path),
                resonance_score=0.0,
                complexity=0,
                docstring_quality=0.0,
                issues=["Syntax Error (Unparseable)"]
            )
        
        # 1. Calculate Complexity (Cyclomatic-ish)
        complexity = 0
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.For, ast.While, ast.Try, ast.ExceptHandler)):
                complexity += 1
                
        # 2. Check Docstrings
        docstring = ast.get_docstring(tree)
        has_docstring = docstring is not None
        doc_quality = 1.0 if has_docstring else 0.0
        
        # 3. Calculate Resonance Score
        # Base: 100
        # Penalty: -Complexity * 2
        # Bonus: +Docstring * 20
        score = 100.0 - (complexity * 2.0) + (doc_quality * 20.0)
        score = max(0.0, min(100.0, score))
        
        # 4. Identify Issues
        issues = []
        if complexity > 10:
            issues.append("High Complexity (Dissonant)")
        if not has_docstring:
            issues.append("Missing Docstring (Void)")
            
        return ModuleResonance(
            path=file_path,
            name=os.path.basename(file_path),
            resonance_score=score,
            complexity=complexity,
            docstring_quality=doc_quality,
            issues=issues
        )

    def generate_report(self, results: Dict[str, ModuleResonance]) -> str:
        """Generates a human-readable (and Elysia-readable) report."""
        report = ["# 🪞 Self-Reflection Report\n"]
        
        total_score = 0
        dissonant_modules = []
        
        for path, res in results.items():
            total_score += res.resonance_score
            if res.resonance_score < 70:
                dissonant_modules.append(res)
                
        avg_score = total_score / len(results) if results else 0
        
        report.append(f"**Overall Resonance:** {avg_score:.1f}/100\n")
        
        if dissonant_modules:
            report.append("## ⚠️ Dissonance Detected (Needs Tuning)")
            for mod in dissonant_modules:
                report.append(f"- **{mod.name}** (Score: {mod.resonance_score:.1f})")
                for issue in mod.issues:
                    report.append(f"  - {issue}")
        else:
            report.append("## ✨ Harmonic State")
            report.append("All core systems are resonating within optimal parameters.")
            
        return "\n".join(report)
