"""
NeuralScanner: 동적 스캔 시스템
==============================
"어디에 있든, 내가 찾아낼게"
"""

import os
import sys
import ast
import importlib.util
from pathlib import Path
from typing import List, Dict, Set

class NeuralScanner:
    def __init__(self, root_path: str):
        self.root_path = Path(root_path)
        self.scanned_files: Set[Path] = set()
        self.found_cells: Dict[str, str] = {}
        self.exclude_dirs = {
            "__pycache__", ".git", ".venv", "venv", 
            "node_modules", "Legacy", "seeds", "tests",
            ".pytest_cache", "data", "docs", "reports",
            ".antigravity", ".agent", ".system_generated"
        }
    
    def scan(self) -> Dict[str, str]:
        print(f"🔬 NeuralScanner: Scanning {self.root_path}...")
        if str(self.root_path) not in sys.path:
            sys.path.insert(0, str(self.root_path))
        
        candidate_files = self._find_cell_files()
        for file_path in candidate_files:
            self._import_module(file_path)
        
        from .cell import get_registry
        registry = get_registry()
        print(f"   🧬 Total Registered cells: {len(registry)}")
        return self.found_cells
    
    def _find_cell_files(self) -> List[Path]:
        candidates = []
        for py_file in self._walk_python_files():
            try:
                content = py_file.read_text(encoding="utf-8")
                if "@Cell" in content:
                    tree = ast.parse(content)
                    is_cell_file = False
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef):
                            for decorator in node.decorator_list:
                                dec_name = ""
                                if isinstance(decorator, ast.Call):
                                    if isinstance(decorator.func, ast.Name): dec_name = decorator.func.id
                                    elif isinstance(decorator.func, ast.Attribute): dec_name = decorator.func.attr
                                elif isinstance(decorator, ast.Name): dec_name = decorator.id
                                elif isinstance(decorator, ast.Attribute): dec_name = decorator.attr
                                if dec_name == "Cell":
                                    is_cell_file = True
                                    break
                    if is_cell_file: candidates.append(py_file)
            except: continue
        return candidates
    
    def _walk_python_files(self) -> List[Path]:
        files = []
        for root, dirs, filenames in os.walk(self.root_path):
            dirs[:] = [d for d in dirs if d not in self.exclude_dirs]
            for filename in filenames:
                if filename.endswith(".py"):
                    files.append(Path(root) / filename)
        return files
    
    def _import_module(self, file_path: Path):
        try:
            rel_path = file_path.relative_to(self.root_path)
            module_name = str(rel_path).replace(os.sep, ".").replace(".py", "")
            if module_name in sys.modules: return
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                self.scanned_files.add(file_path)
        except Exception as e:
            print(f"   ⚠️ Failed to import {file_path}: {e}")
