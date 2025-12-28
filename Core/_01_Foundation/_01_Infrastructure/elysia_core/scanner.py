"""
NeuralScanner: ?숈쟻 ?ㅼ틪 ?쒖뒪??
==============================
"?대뵒???덈뱺, ?닿? 李얠븘?쇨쾶"

?꾨줈洹몃옩 ?쒖옉 ???꾩껜 ?꾨줈?앺듃瑜??ㅼ틪?섏뿬
@Cell ?곗퐫?덉씠?곌? 遺숈? 紐⑤뱺 ?대옒?ㅻ? 李얠븘 ?깅줉?⑸땲??
"""

import os
import sys
import ast
import importlib.util
from pathlib import Path
from typing import List, Dict, Set


class NeuralScanner:
    """
    ?꾩껜 ?꾨줈?앺듃瑜??ㅼ틪?섏뿬 @Cell ?곗퐫?덉씠?곌? 遺숈? ?대옒?ㅻ? 李얠뒿?덈떎.
    """
    
    def __init__(self, root_path: str):
        self.root_path = Path(root_path)
        self.scanned_files: Set[Path] = set()
        self.found_cells: Dict[str, str] = {}  # {identity: file_path}
        
        # ?ㅼ틪 ?쒖쇅 ?대뜑
        self.exclude_dirs = {
            "__pycache__", ".git", ".venv", "venv", 
            "node_modules", "Legacy", "seeds", "tests",
            ".pytest_cache", "data", "docs", "reports",
            ".antigravity", ".agent", ".system_generated"
        }
    
    def scan(self) -> Dict[str, str]:
        """
        ?꾩껜 ?꾨줈?앺듃 ?ㅼ틪
        
        Returns:
            {identity: file_path} ?뺤뀛?덈━
        """
        print(f"?뵮 NeuralScanner: Scanning {self.root_path}...")
        
        # Python 寃쎈줈??猷⑦듃 異붽?
        if str(self.root_path) not in sys.path:
            sys.path.insert(0, str(self.root_path))
        
        # Step 1: @Cell ?곗퐫?덉씠?곌? ?덈뒗 ?뚯씪 李얘린 (AST濡?鍮좊Ⅴ寃?
        candidate_files = self._find_cell_files()
        print(f"   ?뱛 Found {len(candidate_files)} candidate files with @Cell decorator")
        
        # Step 2: ?대떦 ?뚯씪?ㅻ쭔 ?ㅼ젣 ?꾪룷??
        for file_path in candidate_files:
            self._import_module(file_path)
        
        # Step 3: 寃곌낵 諛섑솚
        from .cell import get_registry
        registry = get_registry()
        
        print(f"   ?㎚ Total Registered cells: {len(registry)}")
        for identity in sorted(registry.keys()):
            print(f"      ??{identity}")
        
        return {identity: str(file_path) for identity, file_path in self.found_cells.items()}
    
    def _find_cell_files(self) -> List[Path]:
        """@Cell ?곗퐫?덉씠?곌? ?덈뒗 ?뚯씪留?李얘린 (AST 湲곕컲)"""
        candidates = []
        
        for py_file in self._walk_python_files():
            try:
                print(f"      - Checking {py_file}...")
                content = py_file.read_text(encoding="utf-8")
                if "@Cell" in content:  # 鍮좊Ⅸ ?꾪꽣
                    # AST濡??뺥솗???뺤씤
                    tree = ast.parse(content)
                    is_cell_file = False
                    for node in ast.walk(tree):
                        if isinstance(node, ast.ClassDef):
                            for decorator in node.decorator_list:
                                # @Cell(...) ?먮뒗 @Cell
                                # ?뱀? elysia_core.Cell(...)
                                dec_name = ""
                                if isinstance(decorator, ast.Call):
                                    if isinstance(decorator.func, ast.Name):
                                        dec_name = decorator.func.id
                                    elif isinstance(decorator.func, ast.Attribute):
                                        dec_name = decorator.func.attr
                                elif isinstance(decorator, ast.Name):
                                    dec_name = decorator.id
                                elif isinstance(decorator, ast.Attribute):
                                    dec_name = decorator.attr
                                    
                                if dec_name == "Cell":
                                    is_cell_file = True
                                    break
                        if is_cell_file:
                            break
                    if is_cell_file:
                        candidates.append(py_file)
            except Exception as e:
                # print(f"      - AST Parse Error in {py_file}: {e}")
                continue  # ?뚯떛 ?ㅽ뙣 ??臾댁떆
        
        return candidates
    
    def _walk_python_files(self) -> List[Path]:
        """?꾨줈?앺듃 ??紐⑤뱺 .py ?뚯씪 ?쒗쉶"""
        files = []
        
        for root, dirs, filenames in os.walk(self.root_path):
            # ?쒖쇅 ?대뜑 嫄대꼫?곌린
            dirs[:] = [d for d in dirs if d not in self.exclude_dirs]
            
            for filename in filenames:
                if filename.endswith(".py"):
                    files.append(Path(root) / filename)
        
        return files
    
    def _import_module(self, file_path: Path):
        """?뚯씪??紐⑤뱢濡??꾪룷?명븯??@Cell ?곗퐫?덉씠???ㅽ뻾"""
        try:
            # ?곷? 寃쎈줈濡?紐⑤뱢紐??앹꽦
            rel_path = file_path.relative_to(self.root_path)
            module_name = str(rel_path).replace(os.sep, ".").replace(".py", "")
            
            # ?대? ?꾪룷?몃맂 寃쎌슦 ?ㅽ궢
            if module_name in sys.modules:
                return
            
            # ?숈쟻 ?꾪룷??
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec and spec.loader:
                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                spec.loader.exec_module(module)
                self.scanned_files.add(file_path)
                
        except Exception as e:
            # 媛쒕퀎 ?뚯씪 ?ㅽ뙣 ???꾩껜瑜?以묐떒?섏? ?딆쓬
            print(f"   ?좑툘 Failed to import {file_path}: {e}")
            import traceback
            traceback.print_exc()

