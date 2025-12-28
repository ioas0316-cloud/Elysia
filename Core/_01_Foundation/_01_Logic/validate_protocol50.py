"""
Protocol-50 Validator
=====================
새로운 코드가 HyperQubit 아키텍처를 준수하는지 자동 검증.
구시대적 패턴을 찾아서 경고.
"""

import ast
import re
from pathlib import Path
from typing import List, Dict, Tuple
from dataclasses import dataclass

@dataclass
class Violation:
    """프로토콜 위반"""
    file: str
    line: int
    severity: str  # "ERROR", "WARNING"
    rule: str
    message: str
    old_code: str
    suggested_fix: str

class Protocol50Validator:
    """PROTO-50 준수 검증기"""
    
    def __init__(self):
        self.violations: List[Violation] = []
    
    def validate_file(self, filepath: str) -> List[Violation]:
        """파일 검증"""
        self.violations = []
        
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            lines = content.split('\n')
        
        # Pattern-based검사
        self._check_flat_vectors(lines, filepath)
        self._check_if_else_chains(lines, filepath)
        self._check_string_messages(lines, filepath)
        self._check_activation_pattern(lines, filepath)
        
        # AST-based 검사
        try:
            tree = ast.parse(content)
            self._check_ast_patterns(tree, filepath, lines)
        except SyntaxError:
            pass  # Skip files with syntax errors
        
        return self.violations
    
    def _check_flat_vectors(self, lines: List[str], filepath: str):
        """3D flat vector 사용 검사"""
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
                    message="Flat 3D vector 사용 금지. HyperQubit 사용 필요.",
                    old_code=line.strip(),
                    suggested_fix="# Use HyperQubit instead:\nfrom Core._01_Foundation._05_Governance.Foundation.Mind.hyper_qubit import HyperQubit\nqubit = HyperQubit(name='ConceptName')"
                ))
    
    def _check_if_else_chains(self, lines: List[str], filepath: str):
        """if/elif 체인 검사 (Spectral Routing 권장)"""
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
                if chain_length >= 3:  # 3개 이상의 분기
                    self.violations.append(Violation(
                        file=filepath,
                        line=chain_start,
                        severity="WARNING",
                        rule="PROTO-50.1.2",
                        message=f"if/elif 체인 ({chain_length}개 분기) 발견. Spectral Routing 고려.",
                        old_code=f"Lines {chain_start}-{i}",
                        suggested_fix="# Use Spectral Routing:\nresonances = {cortex_id: cortex.resonate(input) for cortex_id, cortex in cortexes.items()}\nwinner = max(resonances, key=resonances.get)"
                    ))
                chain_start = None
                chain_length = 0
    
    def _check_string_messages(self, lines: List[str], filepath: str):
        """String-based 메시지 검사"""
        # Pattern: outbox.append("...")
        pattern = r'outbox\.append\s*\(\s*["\']'
        
        for i, line in enumerate(lines, 1):
            if re.search(pattern, line):
                self.violations.append(Violation(
                    file=filepath,
                    line=i,
                    severity="WARNING",
                    rule="PROTO-50.3.1",
                    message="String message 사용. FrequencyWave 권장.",
                    old_code=line.strip(),
                    suggested_fix="# Use FrequencyWave:\nfrom Core._01_Foundation._05_Governance.Foundation.Mind.tensor_wave import FrequencyWave, SoulTensor\nwave = FrequencyWave(frequency=50.0, amplitude=1.0)\noutbox.append(SoulTensor(wave=wave))"
                ))
    
    def _check_activation_pattern(self, lines: List[str], filepath: str):
        """node.activation 패턴 검사 (HyperQubit는 probabilities 사용)"""
        pattern = r'\.activation\s*[><=]'
        
        for i, line in enumerate(lines, 1):
            if re.search(pattern, line) and 'HyperQubit' not in line:
                self.violations.append(Violation(
                    file=filepath,
                    line=i,
                    severity="ERROR",
                    rule="PROTO-50.1.1",
                    message="단일 activation 값 사용. HyperQubit probabilities 필요.",
                    old_code=line.strip(),
                    suggested_fix="# Use quantum superposition:\nprobs = qubit.state.probabilities()\ntotal_activation = sum(probs.values())"
                ))
    
    def _check_ast_patterns(self, tree: ast.AST, filepath: str, lines: List[str]):
        """AST 기반 패턴 검사"""
        for node in ast.walk(tree):
            # Dict literal 지식 구조 검사
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
                            message="Flat dictionary 지식 구조. WorldTree 고려.",
                            old_code=line.strip()[:60] + "...",
                            suggested_fix="# Use WorldTree:\nfrom Legacy.Project_Sophia.world_tree import WorldTree\ntree = WorldTree()\nroot = tree.add_seed('ConceptName')"
                        ))
    
    def generate_report(self) -> str:
        """검증 리포트 생성"""
        if not self.violations:
            return "✅ No Protocol-50 violations found!"
        
        report = f"⚠️ Found {len(self.violations)} Protocol-50 violations:\n\n"
        
        # Group by severity
        errors = [v for v in self.violations if v.severity == "ERROR"]
        warnings = [v for v in self.violations if v.severity == "WARNING"]
        
        if errors:
            report += f"🔴 ERRORS ({len(errors)}):\n"
            for v in errors:
                report += f"\n  File: {v.file}:{v.line}\n"
                report += f"  Rule: {v.rule}\n"
                report += f"  Issue: {v.message}\n"
                report += f"  Code: {v.old_code}\n"
                report += f"  Fix:\n{v.suggested_fix}\n"
                report += "  " + "-" * 60 + "\n"
        
        if warnings:
            report += f"\n⚠️  WARNINGS ({len(warnings)}):\n"
            for v in warnings:
                report += f"\n  File: {v.file}:{v.line}\n"
                report += f"  Rule: {v.rule}\n"
                report += f"  Issue: {v.message}\n"
        
        return report


def validate_codebase(root_dir: str = "c:/Elysia/Core"):
    """전체 Core 디렉토리 검증"""
    validator = Protocol50Validator()
    all_violations = []
    
    # Python 파일만 검증
    for filepath in Path(root_dir).rglob("*.py"):
        violations = validator.validate_file(str(filepath))
        all_violations.extend(violations)
    
    validator.violations = all_violations
    return validator.generate_report()


if __name__ == "__main__":
    print("=== Protocol-50 Validation ===\n")
    report = validate_codebase()
    print(report)
    
    print("\n📖 See Legacy/ELYSIAS_PROTOCOL/50_HYPER_QUATERNION_ARCHITECTURE.md for details")
