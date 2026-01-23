"""
Protocol-50 Validator
=====================
        HyperQubit                  .
               .
"""

import ast
import re
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass

@dataclass
class Violation:
    """       """
    file: str
    line: int
    severity: str  # "ERROR", "WARNING"
    rule: str
    message: str
    old_code: str
    suggested_fix: str

class Protocol50Validator:
    """PROTO-50       """
    
    def __init__(self):
        self.violations: List[Violation] = []
    
    def validate_file(self, filepath: str) -> List[Violation]:
        """     """
        self.violations = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')
        
        # Pattern-based  
        self._check_flat_vectors(lines, filepath)
        self._check_if_else_chains(lines, filepath)
        self._check_string_messages(lines, filepath)
        self._check_activation_pattern(lines, filepath)
        
        # AST-based   
        try:
            tree = ast.parse(content)
            self._check_ast_patterns(tree, filepath, lines)
        except SyntaxError:
            pass  # Skip files with syntax errors
        
        return self.violations
    
    def _check_flat_vectors(self, lines: List[str], filepath: str):
        """3D flat vector      """
        pattern = r'np\.array\(\s*\[.*?\]\s*\)'
        
        for i, line in enumerate(lines, 1):
            if re.search(pattern, line):
                # Allow in legacy or test files
                if 'legacy' in filepath.lower() or 'test' in filepath.lower():
                    continue
                
                self.violations.append(Violation(
                    file=filepath,
                    line=i,
                    severity="ERROR",
                    rule="PROTO-50.1.1",
                    message="Flat 3D vector      . HyperQubit      .",
                    old_code=line.strip(),
                    suggested_fix="# Use HyperQubit instead:\nfrom Core.L1_Foundation.Foundation.Mind.hyper_qubit import HyperQubit\nqubit = HyperQubit(name='ConceptName')"
                ))
    
    def _check_if_else_chains(self, lines: List[str], filepath: str):
        """if/elif       (Spectral Routing   )"""
        # Look for intent-based routing patterns
        if_elif_pattern = r'^\s*(if|elif)\s+.*intent.*=='
        
        chain_start = None
        chain_length = 0
        
        for i, line in enumerate(lines, 1):
            if re.search(if_elif_pattern, line):
                if chain_start is None:
                    chain_start = i
                chain_length += 1
            elif chain_start and 'else:' in line:
                chain_length += 1
            else:
                if chain_length >= 3:  # 3        
                    self.violations.append(Violation(
                        file=filepath,
                        line=chain_start,
                        severity="WARNING",
                        rule="PROTO-50.1.2",
                        message=f"if/elif    ({chain_length}    )   . Spectral Routing   .",
                        old_code=f"Lines {chain_start}-{i}",
                        suggested_fix="# Use Spectral Routing:\nresonances = {cortex_id: cortex.resonate(input) for cortex_id, cortex in cortexes.items()}\nwinner = max(resonances, key=resonances.get)"
                    ))
                chain_start = None
                chain_length = 0
    
    def _check_string_messages(self, lines: List[str], filepath: str):
        """String-based       """
        # Pattern: outbox.append("...")
        pattern = r'outbox\.append\s*\(\s*["\']'
        
        for i, line in enumerate(lines, 1):
            if re.search(pattern, line):
                self.violations.append(Violation(
                    file=filepath,
                    line=i,
                    severity="WARNING",
                    rule="PROTO-50.3.1",
                    message="String message   . FrequencyWave   .",
                    old_code=line.strip(),
                    suggested_fix="# Use FrequencyWave:\nfrom Core.L1_Foundation.Foundation.Mind.tensor_wave import FrequencyWave, SoulTensor\nwave = FrequencyWave(frequency=50.0, amplitude=1.0)\noutbox.append(SoulTensor(wave=wave))"
                ))
    
    def _check_activation_pattern(self, lines: List[str], filepath: str):
        """node.activation       (HyperQubit  probabilities   )"""
        pattern = r'\.activation\s*[><=]'
        
        for i, line in enumerate(lines, 1):
            if re.search(pattern, line) and 'HyperQubit' not in line:
                self.violations.append(Violation(
                    file=filepath,
                    line=i,
                    severity="ERROR",
                    rule="PROTO-50.1.1",
                    message="   activation     . HyperQubit probabilities   .",
                    old_code=line.strip(),
                    suggested_fix="# Use quantum superposition:\nprobs = qubit.state.probabilities()\ntotal_activation = sum(probs.values())"
                ))
    
    def _check_ast_patterns(self, tree: ast.AST, filepath: str, lines: List[str]):
        """AST         """
        for node in ast.walk(tree):
            # Dict literal         
            if isinstance(node, ast.Dict):
                # Check for knowledge-like structures
                if hasattr(node, 'lineno') and node.lineno < len(lines):
                    line = lines[node.lineno - 1]
                    if 'knowledge' in line.lower() or 'concepts' in line.lower():
                        self.violations.append(Violation(
                            file=filepath,
                            line=node.lineno,
                            severity="WARNING",
                            rule="PROTO-50.2.2",
                            message="Flat dictionary      . WorldTree   .",
                            old_code=line.strip()[:60] + "...",
                            suggested_fix="# Use WorldTree:\nfrom Legacy.Project_Sophia.world_tree import WorldTree\ntree = WorldTree()\nroot = tree.add_seed('ConceptName')"
                        ))
    
    def generate_report(self) -> str:
        """         """
        if not self.violations:
            return "  No Protocol-50 violations found!"
        
        report = f"   Found {len(self.violations)} Protocol-50 violations:\n\n"
        
        # Group by severity
        errors = [v for v in self.violations if v.severity == "ERROR"]
        warnings = [v for v in self.violations if v.severity == "WARNING"]
        
        if errors:
            report += f"  ERRORS ({len(errors)}):\n"
            for v in errors:
                report += f"\n  File: {v.file}:{v.line}\n"
                report += f"  Rule: {v.rule}\n"
                report += f"  Issue: {v.message}\n"
                report += f"  Code: {v.old_code}\n"
                report += f"  Fix:\n{v.suggested_fix}\n"
                report += "  " + "-" * 60 + "\n"
        
        if warnings:
            report += f"\n    WARNINGS ({len(warnings)}):\n"
            for v in warnings:
                report += f"\n  File: {v.file}:{v.line}\n"
                report += f"  Rule: {v.rule}\n"
                report += f"  Issue: {v.message}\n"
        
        return report


def validate_codebase(root_dir: str = "c:/Elysia/Core"):
    """   Core        """
    validator = Protocol50Validator()
    all_violations = []
    
    # Python       
    for filepath in Path(root_dir).rglob("*.py"):
        violations = validator.validate_file(str(filepath))
        all_violations.extend(violations)
    
    validator.violations = all_violations
    return validator.generate_report()


if __name__ == "__main__":
    print("=== Protocol-50 Validation ===\n")
    report = validate_codebase()
    print(report)
    
    print("\n  See Legacy/ELYSIAS_PROTOCOL/HYPER_QUATERNION_ARCHITECTURE.md for details")