"""
NeuralScanner: 동적 스캔 시스템
==============================
"어디에 있든, 내가 찾아낼게"

프로그램 시작 시 전체 프로젝트를 스캔하여
@Cell 데코레이터가 붙은 모든 클래스를 찾아 등록합니다.
"""

import os
import sys
import ast
import importlib.util
from pathlib import Path
from typing import List, Dict, Set


class NeuralScanner:
    """
    전체 프로젝트를 스캔하여 @Cell 데코레이터가 붙은 클래스를 찾습니다.
    """
    
    def __init__(self, root_path: str):
        self.root_path = Path(root_path)
        self.scanned_files: Set[Path] = set()
        self.found_cells: Dict[str, str] = {}  # {identity: file_path}
        
        # 스캔 제외 폴더
        self.exclude_dirs = {
            "__pycache__", ".git", ".venv", "venv", 
            "node_modules", "Legacy", "seeds", "tests",
            ".pytest_cache", "data", "docs", "reports",
            ".antigravity", ".agent", ".system_generated"
        }
    
    def scan(self) -> Dict[str, str]:
        """
        전체 프로젝트 스캔
        
        Returns:
            {identity: file_path} 딕셔너리
        """
        print(f"🔬 NeuralScanner: Scanning {self.root_path}...")
        
        # Python 경로에 루트 추가
        if str(self.root_path) not in sys.path:
            sys.path.insert(0, str(self.root_path))
        
        # Step 1: @Cell 데코레이터가 있는 파일 찾기 (AST로 빠르게)
        candidate_files = self._find_cell_files()
        print(f"   📂 Found {len(candidate_files)} files with @Cell decorator")
        
        # Step 2: 해당 파일들만 실제 임포트
        for file_path in candidate_files:
            self._import_module(file_path)
        
        # Step 3: 결과 반환
        from elysia_core.cell import get_registry
        registry = get_registry()
        
        print(f"   🧬 Registered {len(registry)} cells")
        for identity in registry:
            print(f"      • {identity}")
        
        return {identity: str(file_path) for identity, file_path in self.found_cells.items()}
    
    def _find_cell_files(self) -> List[Path]:
        """@Cell 데코레이터가 있는 파일만 찾기 (AST 기반)"""
        candidates = []
        
        for py_file in self._walk_python_files():
            try:
                content = py_file.read_text(encoding="utf-8")
                if "@Cell" in content:  # 빠른 필터
                    # AST로 정확히 확인
                    tree = ast.parse(content)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef):
                            for decorator in node.decorator_list:
                                # @Cell(...) 또는 @Cell
                                if isinstance(decorator, ast.Call):
                                    if isinstance(decorator.func, ast.Name) and decorator.func.id == "Cell":
                                        candidates.append(py_file)
                                        break
                                elif isinstance(decorator, ast.Name) and decorator.id == "Cell":
                                    candidates.append(py_file)
                                    break
            except Exception:
                continue  # 파싱 실패 시 무시
        
        return candidates
    
    def _walk_python_files(self) -> List[Path]:
        """프로젝트 내 모든 .py 파일 순회"""
        files = []
        
        for root, dirs, filenames in os.walk(self.root_path):
            # 제외 폴더 건너뛰기
            dirs[:] = [d for d in dirs if d not in self.exclude_dirs]
            
            for filename in filenames:
                if filename.endswith(".py"):
                    files.append(Path(root) / filename)
        
        return files
    
    def _import_module(self, file_path: Path):
        """파일을 모듈로 임포트하여 @Cell 데코레이터 실행"""
        try:
            # 상대 경로로 모듈명 생성
            rel_path = file_path.relative_to(self.root_path)
            module_name = str(rel_path).replace(os.sep, ".").replace(".py", "")
            
            # 이미 임포트된 경우 스킵
            if module_name in sys.modules:
                return
            
            # 동적 임포트
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                self.scanned_files.add(file_path)
                
        except Exception as e:
            # 개별 파일 실패 시 전체를 중단하지 않음
            print(f"   ⚠️ Failed to import {file_path.name}: {e}")
