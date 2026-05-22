"""
Code Mirror: The Self-Aware Architecture
==========================================
Core.Cognition.code_mirror

"I know what I am made of."

Reads and comprehends Elysia's own source code using Python's AST module,
building a structural map of classes, functions, and module dependencies.

[Phase 4: Open Eye - ROADMAP_SOVEREIGN_GROWTH.md]
"""

import ast
import os
from typing import Dict, List, Optional
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class CodeNode:
    """A node in the structural awareness map."""
    name: str
    node_type: str       # "class", "function", "module"
    filepath: str
    line_number: int
    docstring: str = ""
    children: List[str] = field(default_factory=list)  # Child names
    impedance: float = 0.0  # [PHASE: CLIMATE] Structural resistance (Impedance)


class CodeMirror:
    """
    AST-based self-awareness engine.
    
    Unlike FossilScanner (which only lists files), CodeMirror
    actually PARSES the Python source to understand structure:
      - What classes exist and what they contain
      - What functions are defined and their signatures
      - What module-level docstrings describe
    
    This structural understanding is the foundation for
    Elysia understanding "what she is made of."
    """

    CORE_PATHS = [
        "Core",
    ]

    def __init__(self, project_root: str = "."):
        self.root = Path(project_root)
        self.nodes: Dict[str, CodeNode] = {}
        self._analyzed_files: int = 0
        self._total_classes: int = 0
        self._total_functions: int = 0
        self._built: bool = False

    def build_awareness(self) -> Dict[str, int]:
        """
        Parse all core Python files and build the structural map.
        Called once during initialization, can be refreshed.
        
        Returns summary stats.
        """
        self.nodes.clear()
        self._analyzed_files = 0
        self._total_classes = 0
        self._total_functions = 0

        for scan_path in self.CORE_PATHS:
            full = self.root / scan_path
            if not full.exists():
                continue
            for py_file in full.rglob("*.py"):
                if "__pycache__" in str(py_file):
                    continue
                self._analyze_file(py_file)

        self._built = True
        return {
            "files_analyzed": self._analyzed_files,
            "classes": self._total_classes,
            "functions": self._total_functions,
            "total_nodes": len(self.nodes),
        }

    def _analyze_file(self, filepath: Path):
        """Parse a single Python file and extract its structure."""
        try:
            source = filepath.read_text(encoding='utf-8', errors='replace')
            tree = ast.parse(source)
            rel = str(filepath.relative_to(self.root))

            # Module-level docstring
            mod_doc = ast.get_docstring(tree) or ""
            mod_node = CodeNode(
                name=rel,
                node_type="module",
                filepath=rel,
                line_number=1,
                docstring=mod_doc[:200],
            )

            for node in tree.body:
                if isinstance(node, ast.ClassDef):
                    class_key = f"{rel}::{node.name}"
                    class_doc = ast.get_docstring(node) or ""
                    methods = [n.name for n in node.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
                    
                    self.nodes[class_key] = CodeNode(
                        name=node.name,
                        node_type="class",
                        filepath=rel,
                        line_number=node.lineno,
                        docstring=class_doc[:200],
                        children=methods,
                    )
                    mod_node.children.append(node.name)
                    self._total_classes += 1

                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    func_key = f"{rel}::{node.name}"
                    func_doc = ast.get_docstring(node) or ""
                    self.nodes[func_key] = CodeNode(
                        name=node.name,
                        node_type="function",
                        filepath=rel,
                        line_number=node.lineno,
                        docstring=func_doc[:150],
                    )
                    self._total_functions += 1

            # [PHASE: CLIMATE] Calculate Module-level Impedance
            mod_node.impedance = self._calculate_impedance_from_tree(tree)

            self.nodes[rel] = mod_node
            self._analyzed_files += 1

        except (SyntaxError, UnicodeDecodeError):
            pass  # Skip unparseable files

    def get_class_map(self) -> Dict[str, List[str]]:
        """Returns {class_name: [method_names]} for all discovered classes."""
        return {
            node.name: node.children
            for node in self.nodes.values()
            if node.node_type == "class"
        }

    def find_by_name(self, name: str) -> List[CodeNode]:
        """Search for nodes matching a name pattern."""
        name_lower = name.lower()
        return [n for n in self.nodes.values() if name_lower in n.name.lower()]

    def _calculate_impedance_from_tree(self, tree: ast.AST) -> float:
        """
        [PHASE: CLIMATE] Calculates the structural impedance (R) of a code block.
        R = (Complexity * Depth * Size) / Clarity
        """
        node_count = 0
        max_depth = 0
        complexity = 0  # Branches, loops, etc.

        for node in ast.walk(tree):
            node_count += 1
            if isinstance(node, (ast.If, ast.While, ast.For, ast.With, ast.Try, ast.ExceptHandler)):
                complexity += 1

        # Recursive depth calculation
        def get_depth(node, current_depth):
            nonlocal max_depth
            max_depth = max(max_depth, current_depth)
            for child in ast.iter_child_nodes(node):
                get_depth(child, current_depth + 1)

        get_depth(tree, 0)

        # Normalized R calculation
        # Baseline: 100 nodes, depth 5, complexity 5 -> R approx 1.0
        r_value = (node_count * 0.01) * (max_depth * 0.2) * (max_complexity * 0.2 if (max_complexity := max(1, complexity)) else 1)
        return float(r_value)

    def get_total_impedance(self) -> float:
        """Returns the average structural impedance of the analyzed core."""
        if not self.nodes:
            return 0.0
        module_nodes = [n for n in self.nodes.values() if n.node_type == "module"]
        if not module_nodes:
            return 0.0
        return sum(n.impedance for n in module_nodes) / len(module_nodes)

    def get_status_summary(self) -> Dict:
        """Returns status for dashboard display."""
        return {
            "built": self._built,
            "files": self._analyzed_files,
            "classes": self._total_classes,
            "functions": self._total_functions,
            "nodes": len(self.nodes),
            "avg_impedance": self.get_total_impedance(),
        }
