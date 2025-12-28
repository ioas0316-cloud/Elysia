import os
import ast
from pathlib import Path

ROOT = Path(r'C:\Elysia\Core')
results = []

for py_file in ROOT.rglob("*.py"):
    try:
        content = py_file.read_text(encoding='utf-8')
        if '@Cell' in content:
            tree = ast.parse(content)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    for dec in node.decorator_list:
                        name = ""
                        if isinstance(dec, ast.Call) and isinstance(dec.func, ast.Name): name = dec.func.id
                        elif isinstance(dec, ast.Name): name = dec.id
                        if name == "Cell":
                            if isinstance(dec, ast.Call) and dec.args:
                                identity = dec.args[0].value if hasattr(dec.args[0], 'value') else dec.args[0].s
                                results.append(f"{identity}: {node.name} in {py_file}")
    except: pass

for r in sorted(results): print(r)
