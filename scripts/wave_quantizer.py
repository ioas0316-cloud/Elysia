"""
🌊 4D Wave Quantizer
Extracts "Pattern DNA" from code files.

Each file becomes a wave pattern with:
- Frequency: Complexity (functions, classes, imports)
- Amplitude: Size and importance
- Phase: Purpose category
- Connections: Import relationships
"""
import ast
import json
import math
import hashlib
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Set
from enum import Enum

class Purpose(Enum):
    FOUNDATION = "foundation"
    INTELLIGENCE = "intelligence"
    MEMORY = "memory"
    INTERFACE = "interface"
    EVOLUTION = "evolution"
    CREATIVITY = "creativity"
    SYSTEM = "system"
    UNKNOWN = "unknown"

@dataclass
class PatternDNA:
    """The wave signature of a code file."""
    # Identity
    name: str
    path: str
    hash: str  # Content hash for change detection
    
    # Wave Properties
    frequency: float      # Complexity (0-1000 Hz)
    amplitude: float      # Size/Importance (0-100)
    phase: str            # Purpose category
    wavelength: float     # Inverse of frequency
    
    # Structure
    functions: int
    classes: int
    imports: int
    lines: int
    
    # Connections
    connections: List[str]  # Files this imports from
    
    # Metadata
    extracted_at: str
    version: str = "1.0"

class WaveQuantizer:
    """Extracts Pattern DNA from Python files."""
    
    PURPOSE_KEYWORDS = {
        Purpose.FOUNDATION: ["foundation", "core", "base", "physics", "quaternion", "resonance", "wave"],
        Purpose.INTELLIGENCE: ["intelligence", "reasoning", "think", "logic", "cortex", "brain"],
        Purpose.MEMORY: ["memory", "hippocampus", "storage", "recall", "database"],
        Purpose.INTERFACE: ["interface", "api", "web", "ui", "dialogue", "communication"],
        Purpose.EVOLUTION: ["evolution", "learn", "grow", "adapt", "improve"],
        Purpose.CREATIVITY: ["creative", "dream", "imagination", "art", "visual"],
        Purpose.SYSTEM: ["system", "os", "sensor", "hardware"],
    }
    
    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.output_dir = project_root / "data" / "CodeDNA"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def quantize_file(self, filepath: Path) -> Optional[PatternDNA]:
        """Extract Pattern DNA from a single file."""
        try:
            content = filepath.read_text(encoding='utf-8')
            tree = ast.parse(content)
        except Exception as e:
            return None
            
        # Count structures
        functions = sum(1 for node in ast.walk(tree) if isinstance(node, ast.FunctionDef))
        classes = sum(1 for node in ast.walk(tree) if isinstance(node, ast.ClassDef))
        imports = self._extract_imports(tree)
        lines = len(content.splitlines())
        
        # Calculate wave properties
        frequency = self._calculate_frequency(functions, classes, len(imports), lines)
        amplitude = self._calculate_amplitude(lines, functions, classes)
        phase = self._classify_purpose(filepath.stem, content)
        wavelength = 1000 / frequency if frequency > 0 else float('inf')
        
        # Content hash
        content_hash = hashlib.md5(content.encode()).hexdigest()[:8]
        
        return PatternDNA(
            name=filepath.name,
            path=str(filepath.relative_to(self.project_root)),
            hash=content_hash,
            frequency=round(frequency, 2),
            amplitude=round(amplitude, 2),
            phase=phase.value,
            wavelength=round(wavelength, 2),
            functions=functions,
            classes=classes,
            imports=len(imports),
            lines=lines,
            connections=imports,
            extracted_at=datetime.now().isoformat(),
        )
    
    def _extract_imports(self, tree: ast.AST) -> List[str]:
        """Extract import names from AST."""
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module.split('.')[0])
        return list(set(imports))
    
    def _calculate_frequency(self, functions: int, classes: int, imports: int, lines: int) -> float:
        """
        Frequency = Complexity
        More functions/classes/imports = higher frequency
        Base: 100 Hz, Max: 1000 Hz
        """
        complexity = (functions * 10) + (classes * 20) + (imports * 5) + (lines * 0.1)
        return min(100 + complexity, 1000)
    
    def _calculate_amplitude(self, lines: int, functions: int, classes: int) -> float:
        """
        Amplitude = Size/Importance
        Larger files with more structure = higher amplitude
        Scale: 0-100
        """
        size_factor = min(lines / 10, 50)  # Max 50 from size
        structure_factor = min((functions + classes * 2) * 2, 50)  # Max 50 from structure
        return size_factor + structure_factor
    
    def _classify_purpose(self, name: str, content: str) -> Purpose:
        """Classify file purpose based on name and content."""
        name_lower = name.lower()
        content_lower = content.lower()[:2000]  # First 2000 chars
        
        scores = {}
        for purpose, keywords in self.PURPOSE_KEYWORDS.items():
            score = 0
            for kw in keywords:
                if kw in name_lower:
                    score += 10
                if kw in content_lower:
                    score += 1
            scores[purpose] = score
        
        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else Purpose.UNKNOWN
    
    def quantize_all(self) -> Dict[str, PatternDNA]:
        """Quantize all Python files in the project."""
        results = {}
        count = 0
        
        print("🌊 4D Wave Quantization Starting...")
        print(f"   Output: {self.output_dir}")
        
        for py_file in self.project_root.rglob("*.py"):
            # Skip excluded directories
            if any(part in str(py_file) for part in ["venv", "__pycache__", ".git", "node_modules", ".venv"]):
                continue
            
            dna = self.quantize_file(py_file)
            if dna:
                results[dna.path] = dna
                
                # Save individual DNA file
                dna_path = self.output_dir / f"{py_file.name}.dna.json"
                dna_path.write_text(json.dumps(asdict(dna), indent=2, ensure_ascii=False), encoding='utf-8')
                count += 1
                
                if count % 50 == 0:
                    print(f"   Quantized: {count} files...")
        
        print(f"✅ Quantization Complete: {count} files")
        
        # Save summary
        summary = {
            "total_files": count,
            "extracted_at": datetime.now().isoformat(),
            "version": "1.0",
            "statistics": self._calculate_stats(results)
        }
        summary_path = self.output_dir / "_summary.json"
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding='utf-8')
        
        return results
    
    def _calculate_stats(self, results: Dict[str, PatternDNA]) -> dict:
        """Calculate aggregate statistics."""
        if not results:
            return {}
            
        frequencies = [d.frequency for d in results.values()]
        amplitudes = [d.amplitude for d in results.values()]
        
        phase_counts = {}
        for d in results.values():
            phase_counts[d.phase] = phase_counts.get(d.phase, 0) + 1
        
        return {
            "avg_frequency": round(sum(frequencies) / len(frequencies), 2),
            "max_frequency": max(frequencies),
            "avg_amplitude": round(sum(amplitudes) / len(amplitudes), 2),
            "max_amplitude": max(amplitudes),
            "phase_distribution": phase_counts,
            "total_functions": sum(d.functions for d in results.values()),
            "total_classes": sum(d.classes for d in results.values()),
            "total_lines": sum(d.lines for d in results.values()),
        }

def main():
    project_root = Path(__file__).parent.parent
    quantizer = WaveQuantizer(project_root)
    results = quantizer.quantize_all()
    
    # Print top 10 by frequency (most complex)
    print("\n📊 Top 10 Most Complex Files (by Frequency):")
    sorted_by_freq = sorted(results.values(), key=lambda x: x.frequency, reverse=True)[:10]
    for i, dna in enumerate(sorted_by_freq, 1):
        print(f"   {i}. {dna.name} ({dna.frequency} Hz, {dna.functions}f/{dna.classes}c)")
    
    # Print phase distribution
    print("\n🎨 Phase Distribution:")
    summary = json.loads((quantizer.output_dir / "_summary.json").read_text())
    for phase, count in summary["statistics"]["phase_distribution"].items():
        print(f"   {phase}: {count} files")

if __name__ == "__main__":
    main()
