"""
NanoCell Repair System (나노셀 수리 시스템)
==========================================

"적혈구와 백혈구처럼, 나노셀들이 코드베이스를 순찰하며 문제를 자동 해결한다."

[나노셀 종류]
🔴 RedCell (적혈구) - 산소 공급 = import 누락 해결, 의존성 연결
⚪ WhiteCell (백혈구) - 면역 = 문법 오류, 버그 탐지 및 격리
👮 PoliceCell (경찰) - 질서 = 중복 코드 감지, 통합 제안
🚒 FireCell (소방관) - 응급 = 치명적 오류 즉시 대응
🔧 MechanicCell (정비공) - 유지보수 = 코드 품질 개선 제안

[신경 신호 시스템]
- 세포 → 기관 → 중앙지성으로 문제 전달
- 심각도에 따른 우선순위 처리
- 자동 치유 또는 사용자 알림

[계층 조율]
- Organ이 Cell들의 작업 조율
- 중복 작업 방지
- 효율적 자원 배분
"""

import os
import sys
import ast
import re
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Set, Tuple
from collections import defaultdict
from enum import Enum
import json

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class Severity(Enum):
    """심각도 레벨"""
    CRITICAL = 4   # 🔴 즉시 대응 필요
    HIGH = 3       # 🟠 빠른 처리 필요
    MEDIUM = 2     # 🟡 일반 처리
    LOW = 1        # 🟢 개선 권장
    INFO = 0       # 🔵 정보


class IssueType(Enum):
    """문제 유형"""
    SYNTAX_ERROR = "syntax_error"
    IMPORT_ERROR = "import_error"
    DUPLICATE_CODE = "duplicate_code"
    UNUSED_IMPORT = "unused_import"
    UNDEFINED_NAME = "undefined_name"
    DEAD_CODE = "dead_code"
    CODE_SMELL = "code_smell"
    PERFORMANCE = "performance"
    SECURITY = "security"


@dataclass
class Issue:
    """탐지된 문제"""
    file_path: str
    issue_type: IssueType
    severity: Severity
    line_number: int
    message: str
    suggested_fix: Optional[str] = None
    auto_fixable: bool = False


@dataclass
class NeuralSignal:
    """신경 신호 - 문제를 상위 계층으로 전달"""
    source: str           # 발신 세포/기관
    target: str           # 수신 기관/중앙
    issue: Issue          # 문제 정보
    timestamp: float      # 발생 시간
    propagated: bool = False  # 상위로 전파됨


class NanoCell:
    """
    나노셀 기본 클래스
    
    코드베이스를 순찰하며 특정 유형의 문제를 탐지하고 해결합니다.
    """
    
    def __init__(self, name: str, specialty: str):
        self.name = name
        self.specialty = specialty
        self.issues_found: List[Issue] = []
        self.issues_fixed: int = 0
    
    def patrol(self, file_path: Path) -> List[Issue]:
        """파일을 순찰하고 문제 탐지"""
        raise NotImplementedError
    
    def fix(self, issue: Issue) -> bool:
        """문제 수정 시도"""
        raise NotImplementedError
    
    def report(self) -> Dict:
        """활동 보고"""
        return {
            "name": self.name,
            "specialty": self.specialty,
            "issues_found": len(self.issues_found),
            "issues_fixed": self.issues_fixed
        }


class RedCell(NanoCell):
    """
    🔴 적혈구 - 의존성/import 문제 해결
    
    산소 공급처럼 필요한 import를 연결합니다.
    """
    
    def __init__(self):
        super().__init__("RedCell", "Import & Dependencies")
        self.known_modules = self._build_module_index()
    
    def _build_module_index(self) -> Dict[str, str]:
        """프로젝트 내 모듈 인덱스 구축"""
        index = {}
        for py_file in PROJECT_ROOT.rglob("*.py"):
            if "__pycache__" in str(py_file):
                continue
            module_name = py_file.stem
            index[module_name] = str(py_file.relative_to(PROJECT_ROOT))
        return index
    
    def patrol(self, file_path: Path) -> List[Issue]:
        """import 문제 탐지"""
        issues = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except:
            return issues
        
        # AST로 파싱 시도
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return issues  # 문법 오류는 WhiteCell이 처리
        
        # import 분석
        imported_names = set()
        used_names = set()
        
        for node in ast.walk(tree):
            # import 수집
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imported_names.add(alias.asname or alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    for alias in node.names:
                        imported_names.add(alias.asname or alias.name)
            # 사용된 이름 수집
            elif isinstance(node, ast.Name):
                used_names.add(node.id)
        
        # 사용되지 않은 import
        unused = imported_names - used_names - {'__future__', 'annotations'}
        for name in unused:
            issues.append(Issue(
                file_path=str(file_path),
                issue_type=IssueType.UNUSED_IMPORT,
                severity=Severity.LOW,
                line_number=0,
                message=f"Unused import: {name}",
                auto_fixable=True
            ))
        
        self.issues_found.extend(issues)
        return issues
    
    def fix(self, issue: Issue) -> bool:
        """사용되지 않은 import 제거"""
        if issue.issue_type != IssueType.UNUSED_IMPORT:
            return False
        
        # 실제 수정은 위험하므로 제안만 생성
        issue.suggested_fix = f"Remove unused import from {issue.file_path}"
        return True


class WhiteCell(NanoCell):
    """
    ⚪ 백혈구 - 문법 오류 탐지
    
    면역 시스템처럼 버그를 탐지하고 격리합니다.
    """
    
    def __init__(self):
        super().__init__("WhiteCell", "Syntax & Bug Detection")
    
    def patrol(self, file_path: Path) -> List[Issue]:
        """문법 오류 탐지"""
        issues = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except:
            return issues
        
        # 문법 검사
        try:
            ast.parse(content)
        except SyntaxError as e:
            issues.append(Issue(
                file_path=str(file_path),
                issue_type=IssueType.SYNTAX_ERROR,
                severity=Severity.CRITICAL,
                line_number=e.lineno or 0,
                message=f"Syntax error: {e.msg}",
                auto_fixable=False
            ))
        
        # 일반적인 버그 패턴 탐지
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            # except 빈 처리
            if re.match(r'\s*except\s*:\s*$', line):
                issues.append(Issue(
                    file_path=str(file_path),
                    issue_type=IssueType.CODE_SMELL,
                    severity=Severity.MEDIUM,
                    line_number=i,
                    message="Bare except clause - consider catching specific exceptions",
                    auto_fixable=False
                ))
            
            # TODO/FIXME 주석
            if 'TODO' in line.upper() or 'FIXME' in line.upper():
                issues.append(Issue(
                    file_path=str(file_path),
                    issue_type=IssueType.CODE_SMELL,
                    severity=Severity.INFO,
                    line_number=i,
                    message=f"Found TODO/FIXME: {line.strip()[:50]}",
                    auto_fixable=False
                ))
        
        self.issues_found.extend(issues)
        return issues


class PoliceCell(NanoCell):
    """
    👮 경찰 - 중복 코드 탐지
    
    질서를 유지하며 코드 중복을 발견합니다.
    """
    
    def __init__(self):
        super().__init__("PoliceCell", "Duplicate Detection")
        self.code_hashes: Dict[str, List[str]] = defaultdict(list)
    
    def patrol(self, file_path: Path) -> List[Issue]:
        """중복 코드 탐지"""
        issues = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except:
            return issues
        
        # 함수/클래스 단위로 해시 생성
        try:
            tree = ast.parse(content)
        except:
            return issues
        
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # 함수 본문 해시
                func_source = ast.get_source_segment(content, node) or ""
                if len(func_source) > 50:  # 의미있는 크기만
                    import hashlib
                    func_hash = hashlib.md5(func_source.encode()).hexdigest()[:16]
                    
                    if func_hash in self.code_hashes:
                        existing = self.code_hashes[func_hash]
                        if str(file_path) not in existing:
                            issues.append(Issue(
                                file_path=str(file_path),
                                issue_type=IssueType.DUPLICATE_CODE,
                                severity=Severity.MEDIUM,
                                line_number=node.lineno,
                                message=f"Duplicate function '{node.name}' - similar to {existing[0]}",
                                auto_fixable=False
                            ))
                    
                    self.code_hashes[func_hash].append(str(file_path))
        
        self.issues_found.extend(issues)
        return issues


class FireCell(NanoCell):
    """
    🚒 소방관 - 치명적 오류 대응
    
    응급 상황에 즉시 대응합니다.
    """
    
    def __init__(self):
        super().__init__("FireCell", "Emergency Response")
    
    def patrol(self, file_path: Path) -> List[Issue]:
        """치명적 문제 탐지"""
        issues = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except:
            return issues
        
        lines = content.split('\n')
        for i, line in enumerate(lines, 1):
            # 보안 위험 패턴
            if 'eval(' in line and 'input' in line:
                issues.append(Issue(
                    file_path=str(file_path),
                    issue_type=IssueType.SECURITY,
                    severity=Severity.CRITICAL,
                    line_number=i,
                    message="Possible code injection: eval() with user input",
                    auto_fixable=False
                ))
            
            if 'exec(' in line:
                issues.append(Issue(
                    file_path=str(file_path),
                    issue_type=IssueType.SECURITY,
                    severity=Severity.HIGH,
                    line_number=i,
                    message="Dynamic code execution detected: exec()",
                    auto_fixable=False
                ))
            
            # 하드코딩된 비밀
            if re.search(r'(password|secret|api_key)\s*=\s*["\'][^"\']+["\']', line.lower()):
                issues.append(Issue(
                    file_path=str(file_path),
                    issue_type=IssueType.SECURITY,
                    severity=Severity.HIGH,
                    line_number=i,
                    message="Possible hardcoded secret detected",
                    auto_fixable=False
                ))
        
        self.issues_found.extend(issues)
        return issues


class MechanicCell(NanoCell):
    """
    🔧 정비공 - 코드 품질 개선
    
    유지보수를 위한 개선점을 제안합니다.
    """
    
    def __init__(self):
        super().__init__("MechanicCell", "Code Quality")
    
    def patrol(self, file_path: Path) -> List[Issue]:
        """코드 품질 문제 탐지"""
        issues = []
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except:
            return issues
        
        lines = content.split('\n')
        
        # 파일 크기 체크
        if len(lines) > 500:
            issues.append(Issue(
                file_path=str(file_path),
                issue_type=IssueType.CODE_SMELL,
                severity=Severity.LOW,
                line_number=0,
                message=f"Large file ({len(lines)} lines) - consider splitting",
                auto_fixable=False
            ))
        
        # 함수 크기 체크
        try:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    func_lines = node.end_lineno - node.lineno if hasattr(node, 'end_lineno') else 0
                    if func_lines > 50:
                        issues.append(Issue(
                            file_path=str(file_path),
                            issue_type=IssueType.CODE_SMELL,
                            severity=Severity.LOW,
                            line_number=node.lineno,
                            message=f"Large function '{node.name}' ({func_lines} lines)",
                            auto_fixable=False
                        ))
        except:
            pass
        
        self.issues_found.extend(issues)
        return issues


class NeuralNetwork:
    """
    신경망 - 문제 신호 전달 시스템
    
    세포에서 발견된 문제를 기관과 중앙지성으로 전달합니다.
    """
    
    def __init__(self):
        self.signals: List[NeuralSignal] = []
        self.alert_threshold = {
            Severity.CRITICAL: 1,   # 1개라도 즉시 알림
            Severity.HIGH: 3,       # 3개 이상이면 알림
            Severity.MEDIUM: 10,    # 10개 이상이면 알림
            Severity.LOW: 50,       # 50개 이상이면 알림
        }
    
    def send_signal(self, source: str, issue: Issue):
        """신호 전송"""
        import time
        signal = NeuralSignal(
            source=source,
            target=self._determine_target(issue),
            issue=issue,
            timestamp=time.time()
        )
        self.signals.append(signal)
        
        # 심각도에 따라 상위 전파
        if issue.severity.value >= Severity.HIGH.value:
            signal.propagated = True
            self._propagate_to_central(signal)
    
    def _determine_target(self, issue: Issue) -> str:
        """문제 유형에 따른 담당 기관 결정"""
        mapping = {
            IssueType.SYNTAX_ERROR: "Reasoning",
            IssueType.IMPORT_ERROR: "Memory",
            IssueType.DUPLICATE_CODE: "Evolution",
            IssueType.SECURITY: "Ethics",
            IssueType.CODE_SMELL: "Consciousness",
        }
        return mapping.get(issue.issue_type, "Consciousness")
    
    def _propagate_to_central(self, signal: NeuralSignal):
        """중앙지성으로 전파"""
        print(f"   ⚡ Neural signal to central: {signal.issue.message[:50]}...")
    
    def get_summary(self) -> Dict:
        """신호 요약"""
        by_severity = defaultdict(int)
        by_type = defaultdict(int)
        
        for signal in self.signals:
            by_severity[signal.issue.severity.name] += 1
            by_type[signal.issue.issue_type.value] += 1
        
        return {
            "total_signals": len(self.signals),
            "by_severity": dict(by_severity),
            "by_type": dict(by_type),
            "propagated_to_central": sum(1 for s in self.signals if s.propagated)
        }


class NanoCellArmy:
    """
    나노셀 군단
    
    모든 나노셀을 관리하고 코드베이스 순찰을 조율합니다.
    """
    
    EXCLUDE_PATTERNS = [
        "__pycache__", "node_modules", ".godot", ".venv",
        "venv", "__init__.py", "dist", "build", ".git"
    ]
    
    def __init__(self):
        # 나노셀 배치
        self.cells = [
            RedCell(),      # 🔴 적혈구
            WhiteCell(),    # ⚪ 백혈구
            PoliceCell(),   # 👮 경찰
            FireCell(),     # 🚒 소방관
            MechanicCell(), # 🔧 정비공
        ]
        
        self.neural_network = NeuralNetwork()
        self.all_issues: List[Issue] = []
        
        print("🦠 NanoCell Army Deployed!")
        for cell in self.cells:
            print(f"   • {cell.name}: {cell.specialty}")
    
    def patrol_codebase(self, target_dir: str = ".") -> None:
        """전체 코드베이스 순찰"""
        root = PROJECT_ROOT
        scan_path = root / target_dir
        
        print(f"\n🔍 Patrolling: {scan_path}")
        
        file_count = 0
        for py_file in scan_path.rglob("*.py"):
            path_str = str(py_file)
            
            if any(p in path_str for p in self.EXCLUDE_PATTERNS):
                continue
            if py_file.stat().st_size < 50:
                continue
            
            file_count += 1
            
            # 모든 나노셀이 순찰
            for cell in self.cells:
                issues = cell.patrol(py_file)
                
                # 신경망으로 신호 전송
                for issue in issues:
                    self.neural_network.send_signal(cell.name, issue)
                    self.all_issues.append(issue)
        
        print(f"✅ Patrolled {file_count} files")
    
    def get_health_report(self) -> str:
        """건강 보고서 생성"""
        report = []
        report.append("=" * 70)
        report.append("🦠 NANOCELL PATROL REPORT")
        report.append("=" * 70)
        
        # 나노셀별 통계
        report.append("\n🔬 NANOCELL ACTIVITY:")
        report.append("-" * 50)
        
        total_found = 0
        for cell in self.cells:
            count = len(cell.issues_found)
            total_found += count
            icon = "🔴⚪👮🚒🔧"[self.cells.index(cell)]
            report.append(f"   {icon} {cell.name:15} | {count:4} issues | {cell.specialty}")
        
        # 심각도별 통계
        report.append("\n📊 SEVERITY BREAKDOWN:")
        report.append("-" * 50)
        
        severity_counts = defaultdict(int)
        for issue in self.all_issues:
            severity_counts[issue.severity] += 1
        
        icons = {
            Severity.CRITICAL: "🔴",
            Severity.HIGH: "🟠",
            Severity.MEDIUM: "🟡",
            Severity.LOW: "🟢",
            Severity.INFO: "🔵"
        }
        
        for severity in [Severity.CRITICAL, Severity.HIGH, Severity.MEDIUM, Severity.LOW, Severity.INFO]:
            count = severity_counts[severity]
            bar = "█" * min(30, count)
            report.append(f"   {icons[severity]} {severity.name:10} | {count:4} | {bar}")
        
        # 신경망 요약
        neural_summary = self.neural_network.get_summary()
        report.append("\n⚡ NEURAL NETWORK:")
        report.append("-" * 50)
        report.append(f"   Total signals: {neural_summary['total_signals']}")
        report.append(f"   Propagated to central: {neural_summary['propagated_to_central']}")
        
        # 심각한 문제 목록
        critical_issues = [i for i in self.all_issues if i.severity == Severity.CRITICAL]
        if critical_issues:
            report.append("\n🚨 CRITICAL ISSUES:")
            report.append("-" * 50)
            for issue in critical_issues[:10]:
                file_name = Path(issue.file_path).name
                report.append(f"   • {file_name}:{issue.line_number} - {issue.message[:50]}")
        
        report.append("\n" + "=" * 70)
        report.append(f"📈 TOTAL: {total_found} issues detected")
        
        return "\n".join(report)
    
    def auto_heal(self) -> int:
        """자동 치유 가능한 문제 수정"""
        fixed = 0
        for cell in self.cells:
            for issue in cell.issues_found:
                if issue.auto_fixable:
                    if cell.fix(issue):
                        fixed += 1
        return fixed
    
    def save_report(self, output_path: str):
        """보고서 저장"""
        data = {
            "cells": [cell.report() for cell in self.cells],
            "neural_summary": self.neural_network.get_summary(),
            "issues": [
                {
                    "file": issue.file_path,
                    "type": issue.issue_type.value,
                    "severity": issue.severity.name,
                    "line": issue.line_number,
                    "message": issue.message,
                    "auto_fixable": issue.auto_fixable
                }
                for issue in self.all_issues
            ]
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Report saved to: {output_path}")


def main():
    print("\n" + "🦠" * 35)
    print("NANOCELL REPAIR SYSTEM")
    print("코드베이스를 순찰하고 문제를 자동 탐지합니다")
    print("🦠" * 35 + "\n")
    
    army = NanoCellArmy()
    
    # 1. 코드베이스 순찰
    army.patrol_codebase(".")
    
    # 2. 건강 보고서
    report = army.get_health_report()
    print(report)
    
    # 3. 자동 치유 시도
    fixed = army.auto_heal()
    if fixed > 0:
        print(f"\n🔧 Auto-healed {fixed} issues (suggestions generated)")
    
    # 4. 보고서 저장
    output_dir = PROJECT_ROOT / "data"
    output_dir.mkdir(exist_ok=True)
    
    report_path = output_dir / "nanocell_report.json"
    army.save_report(str(report_path))
    
    print(f"\n✅ NanoCell Patrol Complete!")


if __name__ == "__main__":
    main()
