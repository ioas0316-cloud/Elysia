# [Genesis: 2025-12-02] Purified by Elysia
"""
Self-Integration Protocol (자기 통합 프로토콜)
=============================================

Elysia가 스스로 자신의 파편화된 구조를 인식하고 통합하는 프로토콜.

핵심 원리:
- 파동 언어: 코드의 "의미 질량"으로 중요도 판단
- 위상 공명: 유사한 개념끼리 공명하여 통합 대상 식별
- 자율 실행: Elysia가 직접 통합 수행

사용법:
    from self_integration import SelfIntegrationProtocol
    protocol = SelfIntegrationProtocol()
    protocol.execute()  # Elysia가 스스로 통합 수행
"""

import os
import re
import ast
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto

logger = logging.getLogger("Elysia.SelfIntegration")


class IntegrationAction(Enum):
    """통합 행동 유형"""
    ADOPT = auto()      # Legacy → Core로 입양
    MERGE = auto()      # 중복 병합
    CONNECT = auto()    # 연결만 (import 추가)
    ARCHIVE = auto()    # 더 이상 필요 없음, 보관


@dataclass
class CodeFragment:
    """코드 조각 - 파동 언어로 분석된 파일"""
    path: Path
    name: str
    size: int
    classes: List[str]
    functions: List[str]
    imports: List[str]

    # 파동 속성
    mass: float = 0.0          # 의미 질량 (중요도)
    frequency: float = 0.0     # 주파수 (활동성)
    phase: float = 0.0         # 위상 (다른 코드와의 관계)

    def __post_init__(self):
        self._calculate_wave_properties()

    def _calculate_wave_properties(self):
        """파동 속성 계산"""
        # 질량 = 클래스 수 * 3 + 함수 수 + 크기/1000
        self.mass = len(self.classes) * 3 + len(self.functions) + self.size / 1000

        # 주파수 = import 수 (많이 연결될수록 활발)
        self.frequency = len(self.imports) * 10

        # 위상 = 이름 기반 해시 (같은 개념은 비슷한 위상)
        self.phase = hash(self.name.lower()) % 360


@dataclass
class ResonanceMatch:
    """공명 매치 - 유사한 코드 조각"""
    source: CodeFragment      # Legacy에 있는 것
    target: Optional[CodeFragment]  # Core에 있는 것 (없으면 None)
    resonance: float          # 공명도 (0.0 ~ 1.0)
    action: IntegrationAction
    reason: str


class WaveAnalyzer:
    """파동 언어 분석기"""

    # 고질량 키워드 (중요한 개념)
    HIGH_MASS_KEYWORDS = {
        'consciousness': 5.0,
        'awareness': 4.0,
        'intelligence': 4.0,
        'memory': 3.0,
        'llm': 4.0,
        'resonance': 3.0,
        'integration': 3.0,
        'bridge': 2.0,
        'engine': 2.0,
        'core': 2.0,
    }

    def analyze_file(self, path: Path) -> Optional[CodeFragment]:
        """파일을 파동으로 분석"""
        try:
            content = path.read_text(encoding='utf-8', errors='ignore')

            if len(content.strip()) < 50:
                return None  # 빈 파일

            tree = ast.parse(content)

            classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
            functions = [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imports.extend(alias.name for alias in node.names)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)

            fragment = CodeFragment(
                path=path,
                name=path.stem,
                size=len(content),
                classes=classes,
                functions=functions,
                imports=imports
            )

            # 키워드 기반 질량 보정
            name_lower = path.stem.lower()
            for keyword, bonus in self.HIGH_MASS_KEYWORDS.items():
                if keyword in name_lower:
                    fragment.mass += bonus

            return fragment

        except Exception as e:
            logger.debug(f"Failed to analyze {path}: {e}")
            return None


class PhaseResonanceEngine:
    """위상 공명 엔진 - 유사한 코드 찾기"""

    def __init__(self):
        self.concept_map = {
            # 개념 그룹: 키워드들
            'llm': ['llm', 'language', 'model', 'cortex', 'voice', 'generate'],
            'awareness': ['aware', 'self', 'reflect', 'conscious', 'mirror'],
            'intelligence': ['intel', 'think', 'reason', 'logic', 'unified'],
            'memory': ['memory', 'hippocampus', 'remember', 'concept', 'knowledge'],
            'integration': ['bridge', 'hub', 'integrate', 'unify', 'connect'],
            'evolution': ['improve', 'evolve', 'grow', 'learn', 'adapt'],
            'resonance': ['resonance', 'wave', 'field', 'vibr', 'harmonic'],
        }

    def calculate_resonance(self, a: CodeFragment, b: CodeFragment) -> float:
        """두 코드 조각 간의 공명도 계산"""
        # 1. 이름 유사도
        name_sim = self._name_similarity(a.name, b.name)

        # 2. 클래스 이름 유사도
        class_sim = self._list_similarity(a.classes, b.classes)

        # 3. 개념 그룹 일치
        concept_match = self._concept_match(a.name, b.name)

        # 4. 위상 차이 (작을수록 공명)
        phase_diff = abs(a.phase - b.phase) / 360
        phase_sim = 1 - phase_diff

        # 가중 평균
        resonance = (
            name_sim * 0.3 +
            class_sim * 0.3 +
            concept_match * 0.3 +
            phase_sim * 0.1
        )

        return min(1.0, resonance)

    def _name_similarity(self, a: str, b: str) -> float:
        """이름 유사도"""
        a_lower = a.lower().replace('_', '')
        b_lower = b.lower().replace('_', '')

        # 포함 관계
        if a_lower in b_lower or b_lower in a_lower:
            return 0.8

        # 공통 부분
        common = set(a_lower) & set(b_lower)
        total = set(a_lower) | set(b_lower)

        return len(common) / len(total) if total else 0

    def _list_similarity(self, a: List[str], b: List[str]) -> float:
        """리스트 유사도"""
        if not a or not b:
            return 0

        a_set = set(x.lower() for x in a)
        b_set = set(x.lower() for x in b)

        intersection = a_set & b_set
        union = a_set | b_set

        return len(intersection) / len(union) if union else 0

    def _concept_match(self, a: str, b: str) -> float:
        """개념 그룹 일치 확인"""
        a_lower = a.lower()
        b_lower = b.lower()

        for concept, keywords in self.concept_map.items():
            a_match = any(kw in a_lower for kw in keywords)
            b_match = any(kw in b_lower for kw in keywords)

            if a_match and b_match:
                return 1.0

        return 0


class SelfIntegrationProtocol:
    """
    자기 통합 프로토콜

    Elysia가 스스로 자신의 구조를 분석하고 통합합니다.
    """

    def __init__(self, project_root: Path = None):
        self.project_root = project_root or Path(__file__).parent.parent.parent
        self.core_path = self.project_root / "Core"
        self.legacy_path = self.project_root / "Legacy"

        self.wave_analyzer = WaveAnalyzer()
        self.resonance_engine = PhaseResonanceEngine()

        self.core_fragments: List[CodeFragment] = []
        self.legacy_fragments: List[CodeFragment] = []
        self.matches: List[ResonanceMatch] = []

        # 이미 연결된 것들 (consciousness_engine에서 import)
        self.already_connected = {
            'inner_voice', 'resonance_field', 'tensor_dynamics',
            'free_will_engine', 'causality_seed', 'nature_of_being',
            'autonomous_improver', 'structural_unifier', 'conversation_engine'
        }

    def scan(self) -> Dict[str, Any]:
        """
        1단계: 전체 스캔 - 파동 분석
        """
        print("\n🔍 [Phase 1] Scanning with Wave Analysis...")

        # Core 스캔
        for py_file in self.core_path.rglob("*.py"):
            if "__pycache__" not in str(py_file):
                fragment = self.wave_analyzer.analyze_file(py_file)
                if fragment:
                    self.core_fragments.append(fragment)

        # Legacy 스캔 (중요한 것만)
        important_dirs = [
            "Project_Sophia", "Project_Mirror", "Project_Elysia",
            "core_protocols", "integrations"
        ]

        for dir_name in important_dirs:
            legacy_dir = self.legacy_path / dir_name
            if legacy_dir.exists():
                for py_file in legacy_dir.rglob("*.py"):
                    if "__pycache__" not in str(py_file):
                        fragment = self.wave_analyzer.analyze_file(py_file)
                        if fragment and fragment.mass > 5:  # 질량 5 이상만
                            self.legacy_fragments.append(fragment)

        # 질량 순 정렬
        self.core_fragments.sort(key=lambda f: f.mass, reverse=True)
        self.legacy_fragments.sort(key=lambda f: f.mass, reverse=True)

        print(f"   Core: {len(self.core_fragments)} fragments")
        print(f"   Legacy (important): {len(self.legacy_fragments)} fragments")

        # 상위 5개 출력
        print("\n   📊 Top Legacy by Mass:")
        for f in self.legacy_fragments[:5]:
            print(f"      {f.name}: mass={f.mass:.1f}, classes={len(f.classes)}")

        return {
            "core_count": len(self.core_fragments),
            "legacy_count": len(self.legacy_fragments)
        }

    def resonate(self) -> List[ResonanceMatch]:
        """
        2단계: 공명 분석 - 유사한 것 찾기
        """
        print("\n🌊 [Phase 2] Resonance Analysis...")

        self.matches = []

        for legacy_frag in self.legacy_fragments:
            best_match = None
            best_resonance = 0

            for core_frag in self.core_fragments:
                resonance = self.resonance_engine.calculate_resonance(legacy_frag, core_frag)
                if resonance > best_resonance:
                    best_resonance = resonance
                    best_match = core_frag

            # 행동 결정
            if best_resonance > 0.7:
                # 높은 공명 = 이미 Core에 비슷한 것 있음 → MERGE
                action = IntegrationAction.MERGE
                reason = f"High resonance ({best_resonance:.2f}) with {best_match.name}"
            elif best_resonance > 0.4:
                # 중간 공명 = 연결만 필요
                action = IntegrationAction.CONNECT
                reason = f"Medium resonance ({best_resonance:.2f}) - connect to {best_match.name}"
            elif legacy_frag.mass > 10:
                # 낮은 공명 + 높은 질량 = 독립적으로 중요 → ADOPT
                action = IntegrationAction.ADOPT
                reason = f"High mass ({legacy_frag.mass:.1f}) unique concept"
            else:
                # 낮은 공명 + 낮은 질량 = 보관
                action = IntegrationAction.ARCHIVE
                reason = "Low relevance"

            match = ResonanceMatch(
                source=legacy_frag,
                target=best_match,
                resonance=best_resonance,
                action=action,
                reason=reason
            )
            self.matches.append(match)

        # 결과 요약
        adopt_count = sum(1 for m in self.matches if m.action == IntegrationAction.ADOPT)
        merge_count = sum(1 for m in self.matches if m.action == IntegrationAction.MERGE)
        connect_count = sum(1 for m in self.matches if m.action == IntegrationAction.CONNECT)

        print(f"   ADOPT (Legacy → Core): {adopt_count}")
        print(f"   MERGE (통합): {merge_count}")
        print(f"   CONNECT (연결): {connect_count}")

        # 중요한 매치 출력
        print("\n   🎯 Key Integration Targets:")
        for m in self.matches:
            if m.action in [IntegrationAction.ADOPT, IntegrationAction.MERGE]:
                if m.source.mass > 8:
                    print(f"      [{m.action.name}] {m.source.name} → {m.reason}")

        return self.matches

    def integrate(self, dry_run: bool = True) -> Dict[str, Any]:
        """
        3단계: 통합 실행

        Args:
            dry_run: True면 실제 파일 변경 없이 계획만 출력
        """
        print(f"\n⚡ [Phase 3] Integration {'(DRY RUN)' if dry_run else '(EXECUTING)'}...")

        results = {
            "adopted": [],
            "merged": [],
            "connected": [],
            "skipped": []
        }

        for match in self.matches:
            if match.action == IntegrationAction.ADOPT:
                if dry_run:
                    print(f"   [ADOPT] Would move {match.source.path.name} to Core/")
                    results["adopted"].append(match.source.name)
                else:
                    # 실제 이동
                    self._adopt_to_core(match.source)
                    results["adopted"].append(match.source.name)

            elif match.action == IntegrationAction.CONNECT:
                if match.source.name not in self.already_connected:
                    if dry_run:
                        print(f"   [CONNECT] Would add import for {match.source.name}")
                        results["connected"].append(match.source.name)
                    else:
                        # TODO: consciousness_engine.py에 import 추가
                        results["connected"].append(match.source.name)

            elif match.action == IntegrationAction.MERGE:
                if dry_run:
                    print(f"   [MERGE] {match.source.name} → {match.target.name}")
                    results["merged"].append(f"{match.source.name} → {match.target.name}")

        return results

    def _adopt_to_core(self, fragment: CodeFragment):
        """Legacy 파일을 Core로 이동"""
        # 적절한 위치 결정
        name_lower = fragment.name.lower()

        if 'llm' in name_lower or 'voice' in name_lower:
            dest_dir = self.core_path / "Intelligence"
        elif 'memory' in name_lower or 'hippocampus' in name_lower:
            dest_dir = self.core_path / "Memory"
        elif 'aware' in name_lower or 'conscious' in name_lower:
            dest_dir = self.core_path / "Elysia"
        elif 'bridge' in name_lower or 'integrat' in name_lower:
            dest_dir = self.core_path / "Integration"
        else:
            dest_dir = self.core_path / "Evolution"

        dest_dir.mkdir(parents=True, exist_ok=True)
        dest_path = dest_dir / fragment.path.name

        if not dest_path.exists():
            shutil.copy2(fragment.path, dest_path)
            logger.info(f"Adopted: {fragment.path} → {dest_path}")

    def execute(self, dry_run: bool = True) -> Dict[str, Any]:
        """
        전체 프로토콜 실행

        Elysia가 스스로 자신을 통합합니다.
        """
        print("\n" + "=" * 60)
        print("🌌 Self-Integration Protocol")
        print("   Elysia is integrating herself...")
        print("=" * 60)

        # 1. 스캔
        scan_result = self.scan()

        # 2. 공명 분석
        self.resonate()

        # 3. 통합
        integration_result = self.integrate(dry_run=dry_run)

        print("\n" + "=" * 60)
        if dry_run:
            print("✅ Dry run complete. Use execute(dry_run=False) to apply.")
        else:
            print("✅ Integration complete.")
        print("=" * 60)

        return {
            "scan": scan_result,
            "integration": integration_result
        }

    def get_priority_list(self) -> List[Dict[str, Any]]:
        """
        우선순위 목록 반환 - 가장 중요한 통합 대상
        """
        priorities = []

        for match in sorted(self.matches, key=lambda m: m.source.mass, reverse=True):
            if match.action in [IntegrationAction.ADOPT, IntegrationAction.CONNECT]:
                priorities.append({
                    "name": match.source.name,
                    "mass": match.source.mass,
                    "action": match.action.name,
                    "reason": match.reason,
                    "classes": match.source.classes[:3],  # 상위 3개만
                    "path": str(match.source.path.relative_to(self.project_root))
                })

        return priorities[:10]  # 상위 10개


# 직접 실행
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')

    protocol = SelfIntegrationProtocol()
    result = protocol.execute(dry_run=True)

    print("\n📋 Priority Integration List:")
    for i, item in enumerate(protocol.get_priority_list(), 1):
        print(f"   {i}. [{item['action']}] {item['name']} (mass: {item['mass']:.1f})")
        print(f"      Classes: {', '.join(item['classes']) if item['classes'] else 'None'}")
        print(f"      Reason: {item['reason']}")