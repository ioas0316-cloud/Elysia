"""
StructureDocumentGenerator: 구조 분석 결과를 문서로 생성
========================================================

분석 결과를 마크다운 문서, Mermaid 다이어그램, 구조화된 JSON으로 생성합니다.

Usage:
    from Core.02_Intelligence.01_Reasoning.Cognition.structure_document_generator import StructureDocumentGenerator
    
    generator = StructureDocumentGenerator()
    doc = generator.generate_folder_overview("Core/Foundation")
    diagram = generator.generate_connection_map()
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime

# 동일 패키지의 다른 모듈 import
try:
    from Core.02_Intelligence.01_Reasoning.Cognition.codebase_introspector import get_introspector, CodebaseIntrospector
    from Core.02_Intelligence.01_Reasoning.Cognition.why_how_explainer import get_explainer, WhyHowExplainer
except ImportError:
    # 직접 실행 시
    from codebase_introspector import get_introspector, CodebaseIntrospector
    from why_how_explainer import get_explainer, WhyHowExplainer


class StructureDocumentGenerator:
    """
    구조 분석 결과를 문서로 생성하는 생성기
    
    핵심 역할:
    1. 폴더별 개요 문서 생성
    2. 모듈 간 연결 관계 다이어그램 생성
    3. 'Why' 설명 문서 생성
    """
    
    def __init__(self, root_path: Optional[str] = None):
        self.root_path = Path(root_path) if root_path else self._find_project_root()
        self.introspector = get_introspector()
        self.explainer = get_explainer()
        self.reports_path = self.root_path / "reports"
        
    def _find_project_root(self) -> Path:
        """프로젝트 루트를 자동으로 탐지"""
        current = Path(__file__).resolve()
        for _ in range(5):
            if (current / "Core").exists() and (current / "README.md").exists():
                return current
            current = current.parent
        return Path.cwd()
    
    def generate_folder_overview(self, folder: str) -> str:
        """
        폴더별 개요 문서를 마크다운으로 생성합니다.
        
        Args:
            folder: 폴더 경로 (예: "Core/Foundation")
            
        Returns:
            마크다운 형식의 개요 문서
        """
        # 정보 수집
        overview = self.introspector.get_folder_overview(folder)
        why_info = self.explainer.explain_structure_why(folder)
        
        if "error" in overview:
            return f"# Error\n\n{overview['error']}"
        
        # 마크다운 생성
        md = f"""# {folder} 폴더 개요

> 자동 생성일: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## 왜 이 폴더가 존재하는가?

{why_info['why']}

**철학적 근거**: {why_info['philosophy']}

## 통계

| 항목 | 수량 |
|------|------|
| Python 파일 | {overview['python_files']} |
| Markdown 파일 | {overview['markdown_files']} |
| 하위 폴더 | {len(overview['subfolders'])} |

## 하위 폴더

{self._format_list(overview['subfolders']) if overview['subfolders'] else '없음'}

## 주요 모듈

{self._format_list(overview['key_modules']) if overview['key_modules'] else '없음'}

## 관련 프로토콜

{self._format_list(why_info['related_protocols']) if why_info['related_protocols'] else '없음'}

---

*이 문서는 Elysia의 자체 분석 시스템에 의해 생성되었습니다.*
"""
        return md
    
    def generate_connection_map(self) -> str:
        """
        모듈 간 연결 관계를 Mermaid 다이어그램으로 생성합니다.
        
        Returns:
            Mermaid 형식의 다이어그램 코드
        """
        diagram = """```mermaid
graph TB
    subgraph Core["🧠 Core"]
        subgraph Foundation["Foundation (기반)"]
            WavePhysics["WavePhysics"]
            ReasoningEngine["ReasoningEngine"]
            InternalUniverse["InternalUniverse"]
        end
        
        subgraph Intelligence["Intelligence (지능)"]
            WaveCodingSystem["WaveCodingSystem"]
            LogosEngine["LogosEngine"]
            LocalCortex["LocalCortex"]
        end
        
        subgraph Cognition["Cognition (인지)"]
            CognitiveHub["CognitiveHub"]
            WhyEngine["WhyEngine"]
            MetaCognition["MetaCognition"]
        end
        
        subgraph Autonomy["Autonomy (자율)"]
            WaveCoder["WaveCoder"]
            SelfModifier["SelfModifier"]
        end
    end
    
    subgraph Ether["🌐 Ether"]
        GlobalHub["GlobalHub"]
    end
    
    %% 연결선
    GlobalHub --> CognitiveHub
    GlobalHub --> ReasoningEngine
    GlobalHub --> WaveCodingSystem
    
    CognitiveHub --> WhyEngine
    CognitiveHub --> InternalUniverse
    
    ReasoningEngine --> InternalUniverse
    ReasoningEngine --> LogosEngine
    
    WaveCodingSystem --> WaveCoder
    WaveCoder --> SelfModifier
    
    WavePhysics --> WaveCodingSystem
```

## 연결 설명

| 소스 | 타겟 | 연결 방식 |
|------|------|----------|
| GlobalHub | All | 중앙 메시지 버스 (파동 패킷) |
| CognitiveHub | WhyEngine | 개념 이해 → '왜' 분석 |
| ReasoningEngine | InternalUniverse | 사고 결과 저장 |
| WaveCodingSystem | WaveCoder | 파동 분석 → AST 변환 |

*이 다이어그램은 주요 연결만 표시합니다. 실제 시스템은 더 복잡합니다.*
"""
        return diagram
    
    def generate_why_document(self, topic: str) -> str:
        """
        특정 주제에 대한 'Why' 설명 문서를 생성합니다.
        
        Args:
            topic: 주제 (예: "wave", "fractal", "trinity")
            
        Returns:
            마크다운 형식의 설명 문서
        """
        rationale = self.explainer.infer_design_rationale(topic)
        related = self.introspector.find_related_modules(topic)
        
        md = f"""# 왜 '{topic}'인가?

> 자동 생성일: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## 설계 근거

{rationale}

## 관련 모듈

총 {len(related)}개의 모듈이 '{topic}'과 관련되어 있습니다:

{self._format_module_list(related[:10])}

## CODEX 연결

이 설계 원칙은 Elysia의 핵심 철학인 CODEX.md에 기반합니다.

---

*이 문서는 Elysia의 자체 분석 시스템에 의해 생성되었습니다.*
"""
        return md
    
    def generate_full_structure_report(self) -> str:
        """
        전체 프로젝트 구조 보고서를 생성합니다.
        
        Returns:
            마크다운 형식의 전체 보고서
        """
        structure = self.introspector.explore_structure()
        summary = self.introspector.get_connectivity_summary()
        
        md = f"""# Elysia 프로젝트 구조 보고서

> 자동 생성일: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## 프로젝트 개요

| 항목 | 수량 |
|------|------|
| 최상위 폴더 | {len(structure['folders'])} |
| Python 파일 | {structure['file_count']} |

### 파일 확장자별 분포

| 확장자 | 파일 수 |
|--------|---------|
"""
        
        for ext, count in structure['extension_stats'].items():
            md += f"| {ext} | {count} |\n"
        
        md += "\n## 주요 폴더 분석\n\n"
        
        # Core 하위 폴더 분석
        core_folders = ["Foundation", "Intelligence", "Cognition", "Autonomy", "Memory"]
        for folder in core_folders:
            full_path = f"Core/{folder}"
            overview = self.introspector.get_folder_overview(full_path)
            if "error" not in overview:
                md += f"### {full_path}\n\n"
                md += f"- Python 파일: {overview['python_files']}\n"
                md += f"- 하위 폴더: {len(overview['subfolders'])}\n\n"
        
        # CodeDNA 요약
        if summary and "statistics" in summary:
            md += f"""## CodeDNA 통계

| 항목 | 수량 |
|------|------|
| 총 함수 | {summary['statistics']['total_functions']} |
| 총 클래스 | {summary['statistics']['total_classes']} |
| 총 코드 라인 | {summary['statistics']['total_lines']} |

"""
        
        md += """---

*이 보고서는 Elysia의 자체 분석 시스템에 의해 생성되었습니다.*
"""
        return md
    
    def save_report(self, content: str, filename: str) -> Path:
        """
        보고서를 파일로 저장합니다.
        
        Args:
            content: 저장할 내용
            filename: 파일명
            
        Returns:
            저장된 파일 경로
        """
        self.reports_path.mkdir(parents=True, exist_ok=True)
        filepath = self.reports_path / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(content)
        
        return filepath
    
    def _format_list(self, items: List[str]) -> str:
        """리스트를 마크다운 불릿 리스트로 변환"""
        if not items:
            return ""
        return "\n".join(f"- {item}" for item in items)
    
    def _format_module_list(self, modules: List[str]) -> str:
        """모듈 리스트를 마크다운으로 변환"""
        if not modules:
            return "관련 모듈을 찾을 수 없습니다."
        return "\n".join(f"- `{mod}`" for mod in modules)


# 싱글톤 인스턴스
_generator_instance: Optional[StructureDocumentGenerator] = None


def get_generator() -> StructureDocumentGenerator:
    """싱글톤 제너레이터 인스턴스를 반환합니다."""
    global _generator_instance
    if _generator_instance is None:
        _generator_instance = StructureDocumentGenerator()
    return _generator_instance


if __name__ == "__main__":
    # 테스트 실행
    generator = StructureDocumentGenerator()
    
    print("=" * 60)
    print("STRUCTURE DOCUMENT GENERATOR TEST")
    print("=" * 60)
    
    # 폴더 개요 생성
    print("\n📂 Core/Foundation 폴더 개요 생성...")
    overview_doc = generator.generate_folder_overview("Core/Foundation")
    print(overview_doc[:500] + "...")
    
    # 연결 맵 생성
    print("\n🔗 연결 맵 생성...")
    connection_map = generator.generate_connection_map()
    print(connection_map[:300] + "...")
    
    # Why 문서 생성
    print("\n💡 'wave' Why 문서 생성...")
    why_doc = generator.generate_why_document("wave")
    print(why_doc[:400] + "...")
    
    print("\n✅ 테스트 완료!")
