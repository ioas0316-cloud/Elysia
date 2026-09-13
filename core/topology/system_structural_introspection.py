"""
Elysia Core Engine: System Structural Introspection Engine & Dynamic Causal Feeder
=============================================================================
프로젝트 코드베이스의 모든 모듈, 클래스, 함수 및 종속성 관계를 AST(Abstract Syntax Tree)와
정적/동적 종속성 추출 방식으로 스캔하고, 파편화된 코드 구조를 0차 인과 파동,
`RelationalNexusNode`, `ConnectivityBeam` 및 `BitMappedTriggerSubstrate` 위상 데이터로
자가 사영(Self-Assimilation) 및 연속적 피드백(Continuous Feedback Feeder)하는 핵심 엔진입니다.

주요 기능:
1. Global Topological Discovery: 전체 프로젝트 코드베이스 AST 스캔 및 인과 토폴로지 추출
2. Dynamic Introspection Feeder: 코드 구조의 RelationalNexusNode, Causal Field, BitMapped Substrate 결상
3. Structural Introspection Metric: 자가 인식 커버리지(Module Introspection Coverage) 및 미인식 비율(Unmapped Ratio) 계산
4. Architectural Friction & Recursion Control: 무한 메타 재귀 방지(Max Meta Depth, Friction Threshold)
"""

import ast
import os
import sys
import numpy as np
from typing import Dict, List, Set, Any, Optional, Tuple

from core.topology.relational_nexus_node import RelationalNexusNode
from core.memory.bit_mapped_trigger_substrate import BitMappedTriggerSubstrate


class CodeASTVisitor(ast.NodeVisitor):
    """AST Visitor for extracting module definitions, imports, classes, functions, and call relationships."""
    def __init__(self, module_name: str):
        self.module_name = module_name
        self.classes: List[str] = []
        self.functions: List[str] = []
        self.imported_modules: Set[str] = set()
        self.called_functions: Set[str] = set()

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            self.imported_modules.add(alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module:
            self.imported_modules.add(node.module)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef):
        self.classes.append(node.name)
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        self.functions.append(node.name)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        if isinstance(node.func, ast.Name):
            self.called_functions.add(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            self.called_functions.add(node.func.attr)
        self.generic_visit(node)


class SystemStructuralIntrospectionEngine:
    """
    시스템 구조 자가 인지 및 인과 피드백 엔진
    """
    def __init__(self, root_dir: str = ".", max_meta_depth: int = 3, friction_threshold: float = 0.5):
        self.root_dir = os.path.abspath(root_dir)
        self.max_meta_depth = max_meta_depth
        self.friction_threshold = friction_threshold

        self.bit_substrate = BitMappedTriggerSubstrate(num_banks=8, bank_size=64)
        self.discovered_modules: Dict[str, Dict[str, Any]] = {}
        self.nexus_nodes: Dict[str, RelationalNexusNode] = {}
        self.introspected_modules: Set[str] = set()
        self.architectural_friction: float = 0.0

    def scan_codebase_ast(self) -> Dict[str, Any]:
        """
        프로젝트 내 모든 .py 파일을 AST 스캔하여 모듈 정보 및 종속성 그래프 추출
        """
        self.discovered_modules.clear()
        target_dirs = ["core", "synaptic_architecture", "modules", "simulators", "scripts", "mva"]

        for tdir in target_dirs:
            full_tdir = os.path.join(self.root_dir, tdir)
            if not os.path.exists(full_tdir):
                continue
            for root, _, files in os.walk(full_tdir):
                for file in files:
                    if file.endswith(".py") and not file.startswith("__"):
                        filepath = os.path.join(root, file)
                        rel_path = os.path.relpath(filepath, self.root_dir)
                        mod_name = rel_path.replace(os.sep, ".").rstrip(".py")

                        try:
                            with open(filepath, "r", encoding="utf-8") as f:
                                code = f.read()
                            tree = ast.parse(code, filename=filepath)
                            visitor = CodeASTVisitor(mod_name)
                            visitor.visit(tree)

                            self.discovered_modules[mod_name] = {
                                "filepath": rel_path,
                                "classes": visitor.classes,
                                "functions": visitor.functions,
                                "imports": list(visitor.imported_modules),
                                "called_functions": list(visitor.called_functions),
                                "loc": len(code.splitlines()),
                                "size_bytes": len(code.encode("utf-8"))
                            }
                        except Exception as e:
                            # Parse error or encoding issue
                            self.discovered_modules[mod_name] = {
                                "filepath": rel_path,
                                "error": str(e),
                                "loc": 0,
                                "size_bytes": 0
                            }

        total_discovered = len(self.discovered_modules)
        return {
            "total_discovered_modules": total_discovered,
            "modules": self.discovered_modules
        }

    def generate_isomorphic_nexus_nodes(self, depth: int = 1) -> Dict[str, Any]:
        """
        스캔된 모듈 구조를 RelationalNexusNode network 및 Causal Tensor로 위상 동형 매핑
        """
        if depth > self.max_meta_depth:
            return {
                "status": "MAX_META_DEPTH_REACHED",
                "depth": depth,
                "introspected_count": len(self.introspected_modules)
            }

        if not self.discovered_modules:
            self.scan_codebase_ast()

        self.nexus_nodes.clear()
        self.introspected_modules.clear()

        mod_names = list(self.discovered_modules.keys())
        mod_index_map = {name: idx for idx, name in enumerate(mod_names)}

        # Calculate import connections
        in_causal: Dict[str, Dict[str, float]] = {m: {} for m in mod_names}
        out_causal: Dict[str, Dict[str, float]] = {m: {} for m in mod_names}

        unmapped_count = 0
        circular_dependency_count = 0

        for mod_name, info in self.discovered_modules.items():
            if "error" in info:
                unmapped_count += 1
                continue

            self.introspected_modules.add(mod_name)
            imports = info.get("imports", [])
            for imp in imports:
                for target_mod in mod_names:
                    if target_mod != mod_name and (target_mod.endswith(imp) or imp in target_mod):
                        weight = float(min(1.0, 0.1 + (info["loc"] / 1000.0)))
                        out_causal[mod_name][target_mod] = weight
                        in_causal[target_mod][mod_name] = weight

                        if mod_name in out_causal.get(target_mod, {}):
                            circular_dependency_count += 1

        # Create RelationalNexusNode for each module
        for mod_name, info in self.discovered_modules.items():
            node_id = f"NexusNode_{mod_name}"
            loc = info.get("loc", 10)
            classes_count = len(info.get("classes", []))
            functions_count = len(info.get("functions", []))

            payload = {
                "module_name": mod_name,
                "loc": loc,
                "classes_count": classes_count,
                "functions_count": functions_count,
                "filepath": info.get("filepath", "")
            }
            self_schema = {
                "dimension": min(10, max(3, classes_count + functions_count)),
                "stress_limit": float(1.0 + (loc / 500.0))
            }

            raw_bit = np.array([hash(mod_name) % 256, loc % 256, classes_count % 256, functions_count % 256], dtype=np.uint8)
            bit_payload = np.zeros(64, dtype=np.uint8)
            bit_payload[:len(raw_bit)] = raw_bit
            bank_id = hash(mod_name) % 8
            self.bit_substrate.write_payload(bank_id, bit_payload)

            node = RelationalNexusNode(
                node_id=node_id,
                payload=payload,
                self_schema=self_schema,
                in_causal_vectors=in_causal[mod_name],
                out_causal_vectors=out_causal[mod_name],
                relational_tensor=np.eye(min(10, max(3, classes_count + functions_count))) * 0.1,
                bit_substrate=self.bit_substrate
            )
            self.nexus_nodes[node_id] = node

        introspection_coverage = float(len(self.introspected_modules) / max(1, len(self.discovered_modules)))
        unmapped_ratio = float(unmapped_count / max(1, len(self.discovered_modules)))
        self.architectural_friction = float(unmapped_ratio * 2.0 + (circular_dependency_count * 0.05))

        return {
            "total_modules": len(self.discovered_modules),
            "introspected_modules_count": len(self.introspected_modules),
            "nexus_nodes_created": len(self.nexus_nodes),
            "introspection_coverage": introspection_coverage,
            "unmapped_ratio": unmapped_ratio,
            "circular_dependency_count": circular_dependency_count,
            "architectural_friction": self.architectural_friction,
            "is_stable": self.architectural_friction < self.friction_threshold
        }

    def compute_system_causal_field_feedback(self) -> Dict[str, Any]:
        """
        시스템 자기 구조 인지 결과를 Causal Field 피드백 텐서로 계산
        """
        total_stress = sum(node.compute_local_stress() for node in self.nexus_nodes.values())
        avg_curvature = float(np.mean([node.compute_local_curvature() for node in self.nexus_nodes.values()])) if self.nexus_nodes else 0.0

        feedback_tensor = np.array([
            len(self.introspected_modules),
            self.architectural_friction,
            total_stress,
            avg_curvature
        ], dtype=float)

        return {
            "feedback_tensor": feedback_tensor,
            "total_stress": total_stress,
            "average_curvature": avg_curvature,
            "introspection_status": "HIGHLY_SELF_AWARE" if self.architectural_friction < self.friction_threshold else "HIGH_FRICTION_RECONFIGURATION_NEEDED"
        }
