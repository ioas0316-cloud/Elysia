"""
CodebaseIntrospector: Elysia의 자기 코드베이스 탐색 도구
========================================================

Elysia가 자신의 코드베이스를 이해하고 탐색할 수 있게 하는 인트로스펙터.
기존 CodeDNA 시스템과 연동하여 작동합니다.

Usage:
    from Core.Intelligence.Cognition.codebase_introspector import CodebaseIntrospector
    
    introspector = CodebaseIntrospector()
    structure = introspector.explore_structure()
    deps = introspector.analyze_dependencies("Core/Foundation/reasoning_engine.py")
"""

import os
import json
import ast
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ModuleInfo:
    """모듈 정보를 담는 구조체"""
    path: str
    name: str
    purpose: str = ""
    imports: List[str] = field(default_factory=list)
    classes: List[str] = field(default_factory=list)
    functions: List[str] = field(default_factory=list)
    line_count: int = 0
    phase: str = "unknown"
    frequency: float = 100.0


class CodebaseIntrospector:
    """
    Elysia의 자기 코드베이스 탐색 도구
    
    핵심 역할:
    1. 폴더 구조와 파일 분포 분석
    2. 모듈 간 의존성 분석
    3. 개념과 관련된 모듈 찾기
    4. 모듈의 목적 추론
    """
    
    def __init__(self, root_path: Optional[str] = None):
        """
        Args:
            root_path: 프로젝트 루트 경로. None이면 자동 탐지
        """
        self.root_path = Path(root_path) if root_path else self._find_project_root()
        self.codedna_path = self.root_path / "data" / "CodeDNA"
        self._connectivity_cache: Optional[Dict] = None
        self._summary_cache: Optional[Dict] = None
        
    def _find_project_root(self) -> Path:
        """프로젝트 루트를 자동으로 탐지"""
        current = Path(__file__).resolve()
        # Core/Cognition/codebase_introspector.py에서 2단계 위로
        for _ in range(5):
            if (current / "Core").exists() and (current / "README.md").exists():
                return current
            current = current.parent
        return Path.cwd()
    
    def explore_structure(self) -> Dict[str, Any]:
        """
        폴더 구조와 파일 분포를 분석합니다.
        
        Returns:
            {
                "folders": ["Core", "docs", "scripts", ...],
                "file_count": 1494,
                "folder_stats": {
                    "Core": {"files": 751, "subfolders": 47},
                    ...
                },
                "extension_stats": {".py": 1200, ".md": 150, ...}
            }
        """
        folders = []
        folder_stats = {}
        extension_stats = {}
        total_files = 0
        
        # 최상위 폴더 목록
        for item in self.root_path.iterdir():
            if item.is_dir() and not item.name.startswith('.'):
                folders.append(item.name)
                
                # 폴더별 통계
                py_files = list(item.rglob("*.py"))
                subfolders = [d for d in item.rglob("*") if d.is_dir()]
                
                folder_stats[item.name] = {
                    "files": len(py_files),
                    "subfolders": len(subfolders)
                }
                total_files += len(py_files)
        
        # 확장자별 통계
        for ext in [".py", ".md", ".json", ".txt"]:
            count = len(list(self.root_path.rglob(f"*{ext}")))
            if count > 0:
                extension_stats[ext] = count
        
        return {
            "folders": sorted(folders),
            "file_count": total_files,
            "folder_stats": folder_stats,
            "extension_stats": extension_stats,
            "analyzed_at": datetime.now().isoformat()
        }
    
    def analyze_dependencies(self, module_path: str) -> List[str]:
        """
        특정 모듈의 import 의존성을 분석합니다.
        
        Args:
            module_path: 분석할 모듈 경로 (예: "Core/Foundation/reasoning_engine.py")
            
        Returns:
            import된 모듈 목록
        """
        full_path = self.root_path / module_path
        
        if not full_path.exists():
            return []
        
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                source = f.read()
            
            tree = ast.parse(source)
            imports = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name)
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module)
            
            return sorted(set(imports))
            
        except (SyntaxError, UnicodeDecodeError):
            return []
    
    def find_related_modules(self, concept: str) -> List[str]:
        """
        개념과 관련된 모듈을 찾습니다.
        CodeDNA connectivity 데이터를 활용합니다.
        
        Args:
            concept: 찾고자 하는 개념 (예: "reasoning", "wave", "memory")
            
        Returns:
            관련 모듈 경로 목록
        """
        concept_lower = concept.lower()
        related = []
        
        # CodeDNA 연결성 데이터 로드
        connectivity = self._load_connectivity()
        
        if connectivity and "nodes" in connectivity:
            for node in connectivity["nodes"]:
                node_id = node.get("id", "")
                if concept_lower in node_id.lower():
                    related.append(node_id)
        
        # 파일 시스템에서도 직접 검색
        for py_file in self.root_path.rglob("*.py"):
            if concept_lower in py_file.name.lower():
                rel_path = str(py_file.relative_to(self.root_path))
                if rel_path not in related:
                    related.append(rel_path)
        
        return sorted(related)[:20]  # 상위 20개만 반환
    
    def get_module_purpose(self, path: str) -> str:
        """
        모듈의 목적을 docstring과 구조에서 추론합니다.
        
        Args:
            path: 모듈 경로
            
        Returns:
            추론된 목적 설명
        """
        full_path = self.root_path / path
        
        if not full_path.exists():
            return "파일을 찾을 수 없습니다."
        
        try:
            with open(full_path, 'r', encoding='utf-8') as f:
                source = f.read()
            
            tree = ast.parse(source)
            
            # 모듈 docstring 확인
            if (tree.body and isinstance(tree.body[0], ast.Expr) 
                and isinstance(tree.body[0].value, ast.Constant)):
                docstring = tree.body[0].value.value
                # 첫 줄만 반환
                return docstring.split('\n')[0].strip()
            
            # docstring이 없으면 파일명에서 추론
            filename = Path(path).stem
            words = filename.replace('_', ' ').title()
            return f"{words} 관련 모듈"
            
        except (SyntaxError, UnicodeDecodeError):
            return "파싱 오류"
    
    def get_folder_overview(self, folder: str) -> Dict[str, Any]:
        """
        폴더의 개요를 생성합니다.
        
        Args:
            folder: 폴더 경로 (예: "Core/Foundation")
            
        Returns:
            폴더 개요 정보
        """
        folder_path = self.root_path / folder
        
        if not folder_path.exists() or not folder_path.is_dir():
            return {"error": "폴더를 찾을 수 없습니다."}
        
        py_files = list(folder_path.rglob("*.py"))
        md_files = list(folder_path.glob("*.md"))
        subfolders = [d.name for d in folder_path.iterdir() if d.is_dir() and not d.name.startswith('_')]
        
        # README가 있으면 설명 추출
        readme_path = folder_path / "README.md"
        description = ""
        if readme_path.exists():
            with open(readme_path, 'r', encoding='utf-8') as f:
                first_lines = f.read().split('\n')[:5]
                description = ' '.join(first_lines).strip()
        
        return {
            "path": folder,
            "description": description or f"{folder} 폴더",
            "python_files": len(py_files),
            "markdown_files": len(md_files),
            "subfolders": subfolders,
            "key_modules": [f.stem for f in list(folder_path.glob("*.py"))[:10]]
        }
    
    def get_connectivity_summary(self) -> Dict[str, Any]:
        """
        CodeDNA 연결성 요약을 반환합니다.
        """
        summary = self._load_summary()
        
        if not summary:
            return {"error": "CodeDNA 요약을 찾을 수 없습니다."}
        
        return summary
    
    def _load_connectivity(self) -> Optional[Dict]:
        """CodeDNA 연결성 데이터를 로드 (캐싱)"""
        if self._connectivity_cache is not None:
            return self._connectivity_cache
        
        connectivity_file = self.codedna_path / "_connectivity.json"
        if connectivity_file.exists():
            try:
                with open(connectivity_file, 'r', encoding='utf-8') as f:
                    self._connectivity_cache = json.load(f)
                return self._connectivity_cache
            except json.JSONDecodeError:
                pass
        
        return None
    
    def _load_summary(self) -> Optional[Dict]:
        """CodeDNA 요약 데이터를 로드 (캐싱)"""
        if self._summary_cache is not None:
            return self._summary_cache
        
        summary_file = self.codedna_path / "_summary.json"
        if summary_file.exists():
            try:
                with open(summary_file, 'r', encoding='utf-8') as f:
                    self._summary_cache = json.load(f)
                return self._summary_cache
            except json.JSONDecodeError:
                pass
        
        return None


# 싱글톤 인스턴스
_introspector_instance: Optional[CodebaseIntrospector] = None


def get_introspector() -> CodebaseIntrospector:
    """싱글톤 인트로스펙터 인스턴스를 반환합니다."""
    global _introspector_instance
    if _introspector_instance is None:
        _introspector_instance = CodebaseIntrospector()
    return _introspector_instance


if __name__ == "__main__":
    # 테스트 실행
    introspector = CodebaseIntrospector()
    
    print("=" * 60)
    print("CODEBASE INTROSPECTOR TEST")
    print("=" * 60)
    
    # 구조 탐색
    structure = introspector.explore_structure()
    print(f"\n📁 폴더 수: {len(structure['folders'])}")
    print(f"📄 Python 파일 수: {structure['file_count']}")
    print(f"📊 폴더 목록: {structure['folders'][:10]}...")
    
    # 의존성 분석
    deps = introspector.analyze_dependencies("Core/Foundation/reasoning_engine.py")
    print(f"\n🔗 ReasoningEngine 의존성: {deps[:5]}...")
    
    # 관련 모듈 찾기
    related = introspector.find_related_modules("wave")
    print(f"\n🌊 'wave' 관련 모듈: {related[:5]}...")
    
    # CodeDNA 요약
    summary = introspector.get_connectivity_summary()
    if "total_files" in summary:
        print(f"\n📊 CodeDNA 통계:")
        print(f"   총 파일: {summary['total_files']}")
        print(f"   총 함수: {summary['statistics']['total_functions']}")
        print(f"   총 클래스: {summary['statistics']['total_classes']}")
