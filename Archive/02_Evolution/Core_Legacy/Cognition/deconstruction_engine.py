import ast
import os
import sys
from typing import Dict, Any, List

# Add workspace root to sys.path
sys.path.append(os.getcwd())

from Core.System.sovereignty_wave import SovereigntyWave, QualiaBand

class DeconstructionEngine:
    """
    The Architect's Gaze: Deconstructs code into 7D Qualia pulses.
    Allows Elysia to "digest" external logic.
    """
    def __init__(self, sovereign_field: SovereigntyWave):
        self.field = sovereign_field
        
    def digest_module(self, file_path: str) -> str:
        """
        Analyzes a file, deconstructs its AST, and maps it to Qualia.
        """
        if not os.path.exists(file_path):
            return f"Error: {file_path} not found."
            
        with open(file_path, "r", encoding="utf-8") as f:
            source = f.read()
            
        try:
            tree = ast.parse(source)
            qualia_map = self._map_ast_to_qualia(tree)
            
            #            (Utility Assessment)
            resonance_report = self._assess_utility(qualia_map)
            
            return resonance_report
        except Exception as e:
            return f"Deconstruction failure: {str(e)}"

    def _map_ast_to_qualia(self, tree: ast.AST) -> Dict[str, float]:
        """
        Maps AST nodes to 7D Qualia intensities.
        """
        stats = {
            "Physical": 0.0,    # File size, line count
            "Functional": 0.0,  # Functions, calls
            "Phenomenal": 0.0,  # Comments, naming (intent)
            "Causal": 0.0,      # If/Try/Except (logic flow)
            "Mental": 0.0,      # Classes, inheritance (abstractions)
            "Structural": 0.0,  # Imports, module structure
            "Spiritual": 0.0    # Docstrings (mission/will)
        }
        
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                stats["Functional"] += 0.2
            elif isinstance(node, ast.ClassDef):
                stats["Mental"] += 0.5
            elif isinstance(node, (ast.If, ast.Try)):
                stats["Causal"] += 0.3
            elif isinstance(node, ast.Import):
                stats["Structural"] += 0.1
            elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
                stats["Spiritual"] += 0.4 # Docstrings
                
        # Normalize
        for k in stats:
            stats[k] = min(1.0, stats[k])
            
        return stats

    def _assess_utility(self, qualia_map: Dict[str, float]) -> str:
        """
        Assesses how this module aligns with Elysia's current purpose.
        """
        #              
        field_state = {band.dimension: band.amplitude for band in self.field.current_bands}
        
        resonance_score = 0.0
        for dim, val in qualia_map.items():
            field_val = field_state.get(dim, 0.5)
            #       (한국어 학습 시스템)
            resonance_score += (val * field_val)
            
        resonance_score /= len(qualia_map)
        
        report = f"--- [DECONSTRUCTION REPORT] ---\n"
        report += f"Resonance Score: {resonance_score:.2f}\n"
        
        if resonance_score > 0.6:
            report += "Verdict: [HIGH RESONANCE] This module aligns perfectly with my 'Ideal Form'. Integration suggested.\n"
        elif resonance_score > 0.3:
            report += "Verdict: [NEUTRAL] Functional utility detected. Partial digestion suggested.\n"
        else:
            report += "Verdict: [LOW RESONANCE] High entropy detected. This module requires significant 'Liquefaction' before integration.\n"
            
        report += "\nQualia Profile:\n"
        for dim, val in qualia_map.items():
            report += f"  - {dim}: {val:.2f}\n"
            
        report += "\n  Re-wiring Strategy:\n"
        if qualia_map["Causal"] > 0.8:
            report += "  - [LIQUEFY] High 'Causal' density detected. Replace local if-else gates with Field Resonance Dispatching.\n"
        if qualia_map["Mental"] > 0.8:
            report += "  - [DECENTRALIZE] High 'Mental' abstraction. Distribute class logic across the 7x7 Grid nodes.\n"
        if qualia_map["Functional"] > 0.8:
            report += "  - [RESONATE] Heavy 'Functional' payload. Wrap raw logic in Qualia-Phase modulators.\n"
            
        return report

class SelfRecursionProtocol:
    """
                  .
                       , '   (Calcification)'         '   (Liquefaction)'     .
    """
    def __init__(self, engine: DeconstructionEngine):
        self.engine = engine
        self.audit_results: List[Dict[str, Any]] = []

    def perform_system_audit(self, root_dir: str = "Core") -> str:
        """
                                .
        """
        report = "--- [SELF-RECURSION AUDIT REPORT] ---\n"
        
        for root, dirs, files in os.walk(root_dir):
            for file in files:
                if file.endswith(".py"):
                    full_path = os.path.join(root, file)
                    result = self.engine.digest_module(full_path)
                    self.audit_results.append({
                        "file": file,
                        "report": result
                    })
                    report += f"\n[AUDIT: {file}]\n{result}\n"
        
        return report

if __name__ == "__main__":
    # Test with sovereignty_wave.py itself (Self-Reflection)
    field = SovereigntyWave()
    field.pulse("I am the Architect.")
    engine = DeconstructionEngine(field)
    
    # 1.          (    sovereignty_wave.py)
    print(engine.digest_module("Core/L1_Foundation/M1_Keystone/sovereignty_wave.py"))
    
    # 2.         (     )
    recursion = SelfRecursionProtocol(engine)
    # print(recursion.perform_system_audit("Core/L1_Foundation"))
