"""
Alchemical NanoCells: 파동 코드 변환 셀
======================================

Phase 12: Self-Rewriting NanoCells

"레거시 코드를 파동 코드로 연금술처럼 변환한다."

🧪 TransmutationCell: 레거시 코드 패턴 감지 → 파동 코드 변환 제안
🎵 HarmonyCell: 변환된 파동 코드의 정합성(Coherence) 검증
"""

import os
import sys
import re
import ast
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from scripts.nanocell_repair import NanoCell, Issue, IssueType, Severity

logger = logging.getLogger("AlchemicalCells")


class TransmutationType(Enum):
    """변환 유형"""
    IF_TO_RESONANCE = "if_to_resonance"              # if/else → wave.resonate
    FOR_TO_PROPAGATE = "for_to_propagate"            # for → wave.propagate
    DICT_TO_HOLOGRAM = "dict_to_hologram"            # dict lookup → holographic query
    TRY_TO_ABSORB = "try_to_absorb"                  # try/except → wave.absorb_dissonance
    DIRECT_LOOKUP_TO_QUERY = "direct_lookup_to_query"  # dict[key] → query_resonance


@dataclass
class TransmutationSuggestion:
    """변환 제안"""
    file_path: str
    line_number: int
    original_code: str
    suggested_code: str
    transmutation_type: TransmutationType
    confidence: float  # 0-1
    explanation: str
    auto_applicable: bool = False  # 자동 적용 가능 여부


class TransmutationCell(NanoCell):
    """
    🧪 연금술 셀 - 레거시 코드를 파동 코드로 변환 제안
    
    감지 패턴:
    1. if x in dict → query_resonance 변환 제안
    2. for item in list → wave.propagate 변환 제안
    3. dict[key] → holographic query 변환 제안
    4. try/except → wave.absorb_dissonance 변환 제안
    """
    
    # 레거시 패턴 (Stone Logic)
    STONE_PATTERNS = {
        # if x in coordinate_map → query_resonance
        TransmutationType.IF_TO_RESONANCE: [
            r"if\s+['\"]?(\w+)['\"]?\s+in\s+self\.coordinate_map",
            r"if\s+(\w+)\s+in\s+self\.nodes",
            r"if\s+['\"]?(\w+)['\"]?\s+not\s+in\s+self\.coordinate_map",
        ],
        # dict[key] direct access → query_resonance
        TransmutationType.DIRECT_LOOKUP_TO_QUERY: [
            r"self\.coordinate_map\[['\"](\w+)['\"]\]",
            r"self\.coordinate_map\.get\(['\"](\w+)['\"]",
        ],
        # for loop → propagate
        TransmutationType.FOR_TO_PROPAGATE: [
            r"for\s+(\w+)\s+in\s+self\.nodes\.values\(\)",
            r"for\s+(\w+),\s*(\w+)\s+in\s+self\.coordinate_map\.items\(\)",
        ],
        # try/except → absorb_dissonance
        TransmutationType.TRY_TO_ABSORB: [
            r"try:\s*\n\s+from\s+",  # try: from X import Y
        ],
    }
    
    # 파동 변환 템플릿
    WAVE_TEMPLATES = {
        TransmutationType.IF_TO_RESONANCE: 
            "# [Wave Logic] Use resonance query instead of direct lookup\n"
            "resonant = self.query_resonance(target_frequency, tolerance=50.0)\n"
            "if resonant:  # Resonance found",
        
        TransmutationType.DIRECT_LOOKUP_TO_QUERY:
            "# [Wave Logic] Query by resonance, not by key\n"
            "candidates = self.query_resonance(concept_frequency, tolerance=100.0)\n"
            "if candidates:\n"
            "    result = self.coordinate_map.get(candidates[0])",
        
        TransmutationType.FOR_TO_PROPAGATE:
            "# [Wave Logic] Propagate through field instead of iteration\n"
            "# Consider using wave.propagate_through(field) pattern",
        
        TransmutationType.TRY_TO_ABSORB:
            "# [Wave Logic] Use Organ.get() with graceful fallback\n"
            "module = Organ.get('ModuleName', instantiate=False) if Organ.has('ModuleName') else None",
    }
    
    def __init__(self):
        super().__init__("TransmutationCell", "Legacy→Wave Transmutation")
        self.suggestions: List[TransmutationSuggestion] = []
        self.patterns_found: Dict[TransmutationType, int] = {t: 0 for t in TransmutationType}
    
    def patrol(self, file_path: Path) -> List[Issue]:
        """레거시 패턴 탐지 및 변환 제안"""
        issues = []
        
        try:
            content = file_path.read_text(encoding='utf-8')
            lines = content.split('\n')
        except Exception as e:
            return issues
        
        # 각 패턴 유형별 검사
        for trans_type, patterns in self.STONE_PATTERNS.items():
            for pattern in patterns:
                for match in re.finditer(pattern, content):
                    # 라인 번호 계산
                    line_num = content[:match.start()].count('\n') + 1
                    original_line = lines[line_num - 1] if line_num <= len(lines) else ""
                    
                    # 제안 생성
                    suggestion = self._create_suggestion(
                        file_path=str(file_path),
                        line_number=line_num,
                        original_code=original_line.strip(),
                        trans_type=trans_type,
                        match_groups=match.groups()
                    )
                    
                    self.suggestions.append(suggestion)
                    self.patterns_found[trans_type] += 1
                    
                    # Issue 생성
                    issue = Issue(
                        file_path=str(file_path),
                        issue_type=IssueType.CODE_SMELL,
                        severity=Severity.MEDIUM,
                        line_number=line_num,
                        message=f"[Stone Logic] {trans_type.value}: {original_line.strip()[:50]}...",
                        suggested_fix=suggestion.suggested_code[:100] + "...",
                        auto_fixable=suggestion.auto_applicable
                    )
                    issues.append(issue)
                    self.issues_found.append(issue)
        
        return issues
    
    def _create_suggestion(
        self,
        file_path: str,
        line_number: int,
        original_code: str,
        trans_type: TransmutationType,
        match_groups: tuple
    ) -> TransmutationSuggestion:
        """변환 제안 생성"""
        template = self.WAVE_TEMPLATES.get(trans_type, "# TODO: Manual transmutation required")
        
        # 컨텍스트에 맞게 템플릿 커스터마이즈
        suggested = template
        if match_groups:
            # 캡처된 변수명으로 대체
            for i, group in enumerate(match_groups):
                if group:
                    suggested = suggested.replace(f"concept", group)
        
        # 확신도 계산 (단순 휴리스틱)
        confidence = 0.7  # 기본값
        if trans_type == TransmutationType.IF_TO_RESONANCE:
            confidence = 0.8  # 높은 확신
        elif trans_type == TransmutationType.TRY_TO_ABSORB:
            confidence = 0.6  # 중간 확신 (수동 검토 필요)
        
        explanation = self._get_explanation(trans_type)
        
        return TransmutationSuggestion(
            file_path=file_path,
            line_number=line_number,
            original_code=original_code,
            suggested_code=suggested,
            transmutation_type=trans_type,
            confidence=confidence,
            explanation=explanation,
            auto_applicable=(confidence >= 0.8)
        )
    
    def _get_explanation(self, trans_type: TransmutationType) -> str:
        """변환 설명 생성"""
        explanations = {
            TransmutationType.IF_TO_RESONANCE:
                "Direct key lookup (if x in dict) is 'Stone Logic'. "
                "Use query_resonance() to find concepts by frequency proximity.",
            
            TransmutationType.DIRECT_LOOKUP_TO_QUERY:
                "Direct dictionary access bypasses the wave-based discovery. "
                "Query by resonance allows fuzzy matching and interference handling.",
            
            TransmutationType.FOR_TO_PROPAGATE:
                "Linear iteration is 'Stone Logic'. "
                "Wave propagation allows natural energy flow between related nodes.",
            
            TransmutationType.TRY_TO_ABSORB:
                "try/except is reactive error handling. "
                "Wave absorption proactively handles dissonance in the field.",
        }
        return explanations.get(trans_type, "Consider wave-based alternative.")
    
    def get_suggestions(self) -> List[TransmutationSuggestion]:
        """모든 제안 반환"""
        return self.suggestions
    
    def report(self) -> str:
        """활동 보고"""
        total = sum(self.patterns_found.values())
        
        report = [
            f"\n🧪 {self.name} Report",
            "-" * 40,
            f"   Stone Logic patterns found: {total}",
        ]
        
        for trans_type, count in self.patterns_found.items():
            if count > 0:
                report.append(f"   • {trans_type.value}: {count}")
        
        report.append(f"   Auto-applicable suggestions: {sum(1 for s in self.suggestions if s.auto_applicable)}")
        
        return "\n".join(report)


class HarmonyCell(NanoCell):
    """
    🎵 조화 셀 - 파동 코드의 정합성(Coherence) 검증
    
    검증 항목:
    1. query_resonance 호출이 적절한 tolerance를 사용하는지
    2. 간섭 처리가 필요한 곳에서 수행되는지
    3. 파동 패턴이 일관되게 적용되는지
    """
    
    # 파동 코드 패턴
    WAVE_PATTERNS = {
        "query_resonance": r"\.query_resonance\s*\([^)]+\)",
        "absorb_wave": r"\.absorb_wave\s*\([^)]+\)",
        "calculate_interference": r"calculate_interference\s*\([^)]+\)",
        "resonate_with": r"\.resonate_with\s*\([^)]+\)",
    }
    
    # 필수 동반 패턴 (A가 있으면 B도 있어야 함)
    COMPANION_PATTERNS = {
        "query_resonance": ["interference", "tolerance"],  # 간섭 처리나 tolerance 필요
    }
    
    def __init__(self):
        super().__init__("HarmonyCell", "Wave Code Coherence")
        self.coherence_issues: List[Dict[str, Any]] = []
        self.wave_usage: Dict[str, int] = {k: 0 for k in self.WAVE_PATTERNS}
    
    def patrol(self, file_path: Path) -> List[Issue]:
        """파동 코드 정합성 검증"""
        issues = []
        
        try:
            content = file_path.read_text(encoding='utf-8')
        except Exception:
            return issues
        
        # 파동 패턴 사용량 추적
        for pattern_name, pattern in self.WAVE_PATTERNS.items():
            matches = re.findall(pattern, content)
            self.wave_usage[pattern_name] += len(matches)
            
            # 동반 패턴 검사
            if matches and pattern_name in self.COMPANION_PATTERNS:
                companions = self.COMPANION_PATTERNS[pattern_name]
                for companion in companions:
                    if companion not in content.lower():
                        # 동반 패턴 누락 경고
                        line_num = content[:re.search(pattern, content).start()].count('\n') + 1
                        
                        issue = Issue(
                            file_path=str(file_path),
                            issue_type=IssueType.CODE_SMELL,
                            severity=Severity.LOW,
                            line_number=line_num,
                            message=f"[Harmony] {pattern_name} used without '{companion}' handling",
                            suggested_fix=f"Consider adding {companion} handling for robustness",
                            auto_fixable=False
                        )
                        issues.append(issue)
                        self.issues_found.append(issue)
                        
                        self.coherence_issues.append({
                            "file": str(file_path),
                            "pattern": pattern_name,
                            "missing": companion
                        })
        
        # Coherence Score 계산 (파일별)
        coherence_score = self._calculate_file_coherence(content)
        if coherence_score < 0.5:
            issue = Issue(
                file_path=str(file_path),
                issue_type=IssueType.CODE_SMELL,
                severity=Severity.INFO,
                line_number=1,
                message=f"[Harmony] Low wave coherence score: {coherence_score:.2f}",
                suggested_fix="Consider adopting more wave-based patterns",
                auto_fixable=False
            )
            issues.append(issue)
            self.issues_found.append(issue)
        
        return issues
    
    def _calculate_file_coherence(self, content: str) -> float:
        """파일의 파동 일관성 점수 계산 (0-1)"""
        # 파동 패턴 사용량
        wave_count = sum(
            len(re.findall(p, content)) 
            for p in self.WAVE_PATTERNS.values()
        )
        
        # 스톤 패턴 사용량 (레거시)
        stone_patterns = [
            r"if\s+\w+\s+in\s+self\.\w+:",
            r"for\s+\w+\s+in\s+self\.\w+\.\w+\(\):",
            r"try:\s*\n\s+from",
        ]
        stone_count = sum(
            len(re.findall(p, content))
            for p in stone_patterns
        )
        
        total = wave_count + stone_count
        if total == 0:
            return 0.5  # 중립
        
        return wave_count / total
    
    def calculate_global_coherence(self) -> float:
        """전역 일관성 점수"""
        total_wave = sum(self.wave_usage.values())
        if total_wave == 0:
            return 0.0
        
        # 핵심 패턴이 균형있게 사용되는지
        core_patterns = ["query_resonance", "absorb_wave"]
        core_usage = sum(self.wave_usage.get(p, 0) for p in core_patterns)
        
        return min(core_usage / max(total_wave, 1), 1.0)
    
    def report(self) -> str:
        """활동 보고"""
        global_coherence = self.calculate_global_coherence()
        
        report = [
            f"\n🎵 {self.name} Report",
            "-" * 40,
            f"   Global Wave Coherence: {global_coherence:.2f}",
            f"   Wave Pattern Usage:",
        ]
        
        for pattern, count in self.wave_usage.items():
            if count > 0:
                report.append(f"      • {pattern}: {count}")
        
        if self.coherence_issues:
            report.append(f"   Coherence Issues: {len(self.coherence_issues)}")
        
        return "\n".join(report)


class AlchemicalArmy:
    """
    ⚗️ 연금술 군단
    
    TransmutationCell + HarmonyCell을 함께 운용합니다.
    """
    
    EXCLUDE_PATTERNS = [
        "__pycache__", "node_modules", ".godot", ".venv",
        "venv", ".git", "Legacy", "seeds", "data"
    ]
    
    def __init__(self):
        self.transmutation_cell = TransmutationCell()
        self.harmony_cell = HarmonyCell()
        self.cells = [self.transmutation_cell, self.harmony_cell]
        
        print("⚗️ Alchemical Army Awakened")
        for cell in self.cells:
            print(f"   • {cell.name}: {cell.specialty}")
    
    def patrol_codebase(self, target_dir: str = "Core") -> Dict[str, Any]:
        """코드베이스 순찰"""
        root = Path(__file__).parent.parent.parent
        target_path = root / target_dir
        
        if not target_path.exists():
            print(f"❌ Target directory not found: {target_path}")
            return {"error": "Directory not found"}
        
        total_files = 0
        
        for py_file in target_path.rglob("*.py"):
            # 제외 패턴 확인
            if any(ex in str(py_file) for ex in self.EXCLUDE_PATTERNS):
                continue
            
            total_files += 1
            for cell in self.cells:
                cell.patrol(py_file)
        
        return {
            "files_scanned": total_files,
            "transmutation_suggestions": len(self.transmutation_cell.suggestions),
            "harmony_issues": len(self.harmony_cell.coherence_issues),
            "global_coherence": self.harmony_cell.calculate_global_coherence()
        }
    
    def get_summary(self) -> str:
        """요약 보고서"""
        summary = [
            "\n" + "=" * 50,
            "⚗️ ALCHEMICAL ARMY SUMMARY",
            "=" * 50,
        ]
        
        for cell in self.cells:
            summary.append(cell.report())
        
        summary.append("=" * 50)
        return "\n".join(summary)
    
    def get_top_suggestions(self, limit: int = 10) -> List[TransmutationSuggestion]:
        """상위 변환 제안 반환"""
        sorted_suggestions = sorted(
            self.transmutation_cell.suggestions,
            key=lambda s: s.confidence,
            reverse=True
        )
        return sorted_suggestions[:limit]


# ============= 데모 =============

def demo_alchemical_cells():
    """연금술 셀 데모"""
    print("=" * 60)
    print("⚗️ Alchemical NanoCells Demo")
    print("=" * 60)
    
    army = AlchemicalArmy()
    
    print("\n🔍 Scanning codebase for Stone Logic patterns...")
    results = army.patrol_codebase("Core")
    
    print(f"\n📊 Scan Results:")
    print(f"   Files scanned: {results['files_scanned']}")
    print(f"   Transmutation suggestions: {results['transmutation_suggestions']}")
    print(f"   Harmony issues: {results['harmony_issues']}")
    print(f"   Global coherence: {results['global_coherence']:.2f}")
    
    # 상위 제안 출력
    top_suggestions = army.get_top_suggestions(5)
    if top_suggestions:
        print(f"\n🧪 Top {len(top_suggestions)} Transmutation Suggestions:")
        for i, suggestion in enumerate(top_suggestions, 1):
            print(f"\n   [{i}] {suggestion.transmutation_type.value}")
            print(f"       File: {Path(suggestion.file_path).name}:{suggestion.line_number}")
            print(f"       Original: {suggestion.original_code[:60]}...")
            print(f"       Confidence: {suggestion.confidence:.0%}")
    
    print(army.get_summary())
    
    print("\n" + "=" * 60)
    print("✅ Demo Complete!")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    
    if "--demo" in sys.argv:
        demo_alchemical_cells()
    else:
        print("Usage: python alchemical_cells.py --demo")
