"""
OrganicBloodCells: Neural Registry 인식 나노셀
=============================================
기존 RedCell/WhiteCell을 확장하여 Neural Registry 패턴을 인식합니다.

🔴 OrganicRedCell: 레거시 import → Organ.get() 변환 제안
⚪ OrganicWhiteCell: @Cell 데코레이터 누락 탐지
"""

import ast
import re
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.nanocell_repair import (
    NanoCell, Issue, IssueType, Severity
)


class OrganicIssueType(Enum):
    """Neural Registry 관련 문제 유형"""
    LEGACY_IMPORT = "legacy_import"           # 레거시 import 사용
    MISSING_CELL_DECORATOR = "missing_cell"   # @Cell 데코레이터 누락
    ORGAN_AVAILABLE = "organ_available"       # Organ.get()으로 대체 가능


class OrganicRedCell(NanoCell):
    """
    🔴 유기적 적혈구 - 레거시 import 탐지 및 변환 제안
    
    from Core.X.Y import Z → Organ.get("Z") 변환을 감지하고 제안합니다.
    """
    
    # 변환 가능한 모듈 패턴
    CONVERTIBLE_PATTERNS = [
        r"from Core\.\w+\.\w+ import (\w+)",
        r"from Core\.\w+ import (\w+)",
    ]
    
    # 알려진 Cell 정체성 (elysia_core/cells/core_cells.py 기반)
    KNOWN_CELLS = {
        "TorchGraph", "TinyBrain", "UnifiedUnderstanding",
        "CognitiveHub", "Trinity", "Conscience",
        "VisionCortex", "MultimodalBridge", "SelfModifier", "DreamDaemon"
    }
    
    def __init__(self):
        super().__init__("OrganicRedCell", "Legacy Import Detection")
        self.convertible_imports: Dict[str, List[str]] = {}
    
    def patrol(self, file_path: Path) -> List[Issue]:
        """레거시 import 탐지"""
        issues = []
        
        # elysia_core 자체와 tests는 제외
        path_str = str(file_path)
        if "elysia_core" in path_str or "tests" in path_str:
            return issues
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except:
            return issues
        
        lines = content.split('\n')
        
        for i, line in enumerate(lines, 1):
            # Core 모듈에서 import하는 패턴 탐지
            for pattern in self.CONVERTIBLE_PATTERNS:
                match = re.search(pattern, line)
                if match:
                    imported_name = match.group(1)
                    
                    # 알려진 Cell인지 확인
                    if imported_name in self.KNOWN_CELLS:
                        issues.append(Issue(
                            file_path=str(file_path),
                            issue_type=IssueType.IMPORT_ERROR,  # 기존 타입 활용
                            severity=Severity.MEDIUM,
                            line_number=i,
                            message=f"Legacy import '{imported_name}' can be replaced with Organ.get('{imported_name}')",
                            suggested_fix=f"from Core._01_Foundation._01_Infrastructure.elysia_core import Organ\n# ...\n{imported_name.lower()} = Organ.get('{imported_name}')",
                            auto_fixable=False  # 수동 검토 필요
                        ))
        
        self.issues_found.extend(issues)
        return issues


class OrganicWhiteCell(NanoCell):
    """
    ⚪ 유기적 백혈구 - @Cell 데코레이터 누락 탐지
    
    중요한 클래스에 @Cell 데코레이터가 없는지 확인합니다.
    """
    
    # @Cell이 있어야 할 클래스 패턴
    IMPORTANT_CLASS_PATTERNS = [
        r"class (\w+Engine)",      # Engine 류
        r"class (\w+System)",      # System 류
        r"class (\w+Cortex)",      # Cortex 류
        r"class (\w+Hub)",         # Hub 류
        r"class (\w+Bridge)",      # Bridge 류
    ]
    
    def __init__(self):
        super().__init__("OrganicWhiteCell", "Cell Decorator Detection")
    
    def patrol(self, file_path: Path) -> List[Issue]:
        """@Cell 데코레이터 누락 탐지"""
        issues = []
        
        # elysia_core 자체는 제외
        if "elysia_core" in str(file_path):
            return issues
        
        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
        except:
            return issues
        
        # 이미 @Cell이 있는지 확인
        has_cell_decorator = "@Cell" in content
        if has_cell_decorator:
            return issues
        
        # AST로 파싱
        try:
            tree = ast.parse(content)
        except SyntaxError:
            return issues
        
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                class_name = node.name
                
                # 중요한 클래스 패턴인지 확인
                for pattern in self.IMPORTANT_CLASS_PATTERNS:
                    if re.match(pattern, class_name):
                        # 이미 데코레이터가 있는지 확인
                        has_decorator = len(node.decorator_list) > 0
                        
                        issues.append(Issue(
                            file_path=str(file_path),
                            issue_type=IssueType.CODE_SMELL,
                            severity=Severity.LOW,
                            line_number=node.lineno,
                            message=f"Class '{class_name}' could benefit from @Cell decorator for organic registry",
                            suggested_fix=f"from Core._01_Foundation._01_Infrastructure.elysia_core import Cell\n\n@Cell('{class_name}')\nclass {class_name}:",
                            auto_fixable=False
                        ))
                        break
        
        self.issues_found.extend(issues)
        return issues


class OrganicCellArmy:
    """
    🦠 유기적 나노셀 군단
    
    Neural Registry 인식 나노셀들을 관리합니다.
    """
    
    EXCLUDE_PATTERNS = [
        "__pycache__", "node_modules", ".godot", ".venv",
        "venv", "__init__.py", "dist", "build", ".git",
        "Legacy", "seeds", "tests"
    ]
    
    def __init__(self):
        self.cells = [
            OrganicRedCell(),
            OrganicWhiteCell(),
        ]
        print("🦠 Organic Cell Army Deployed!")
        for cell in self.cells:
            print(f"   • {cell.name}: {cell.specialty}")
    
    def patrol_codebase(self, target_dir: str = "Core") -> List[Issue]:
        """코드베이스 순찰"""
        from pathlib import Path
        
        root = Path("c:/Elysia")
        scan_path = root / target_dir
        
        print(f"\n🔍 Organic Patrol: {scan_path}")
        
        all_issues = []
        file_count = 0
        
        for py_file in scan_path.rglob("*.py"):
            path_str = str(py_file)
            
            if any(p in path_str for p in self.EXCLUDE_PATTERNS):
                continue
            if py_file.stat().st_size < 50:
                continue
            
            file_count += 1
            
            for cell in self.cells:
                issues = cell.patrol(py_file)
                all_issues.extend(issues)
        
        print(f"✅ Patrolled {file_count} files, found {len(all_issues)} organic issues")
        
        return all_issues
    
    def get_summary(self) -> str:
        """요약 보고서"""
        lines = ["🦠 ORGANIC CELL PATROL SUMMARY", "-" * 40]
        
        for cell in self.cells:
            count = len(cell.issues_found)
            lines.append(f"   • {cell.name}: {count} issues")
        
        return "\n".join(lines)


def main():
    print("\n" + "🧬" * 30)
    print("ORGANIC NANOCELL PATROL")
    print("Neural Registry 인식 코드베이스 순찰")
    print("🧬" * 30 + "\n")
    
    army = OrganicCellArmy()
    issues = army.patrol_codebase("Core")
    
    print("\n" + army.get_summary())
    
    # 샘플 출력
    if issues:
        print("\n📋 Sample Issues (top 5):")
        for issue in issues[:5]:
            file_name = Path(issue.file_path).name
            print(f"   • {file_name}:{issue.line_number} - {issue.message[:60]}...")


if __name__ == "__main__":
    main()
