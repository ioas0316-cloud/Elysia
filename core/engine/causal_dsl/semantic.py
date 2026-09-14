"""
Semantic analyzer and validator for Causal DSL programs.
Validates memory alignments (8B, 16B, 32B, 64B) and checks for circular causality in rules.
"""

from typing import List, Dict, Set, Tuple
from .ast_nodes import Program

TYPE_SIZES = {
    "uint8": 1, "uint16": 2, "uint32": 4, "uint64": 8,
    "int8": 1, "int16": 2, "int32": 4, "int64": 8,
    "float": 4, "double": 8, "half": 2,
    "half3": 6, "float3": 12, "float4": 16,
    "Matrix3x4": 48, "Matrix4x4": 64
}

class CausalSemanticAnalyzer:
    def __init__(self, program: Program):
        self.program = program

    def validate_alignment_and_packing(self) -> List[str]:
        """
        Validates signal size and memory alignment constraints.
        """
        errors = []
        for signal in self.program.signals:
            declared_size = signal.attrs.get("size", "8B")
            size_val = int(declared_size.rstrip("B")) if declared_size.endswith("B") else 8

            total_size = 0
            for m in signal.members:
                sz = TYPE_SIZES.get(m.type_spec, 4)
                total_size += sz

            if total_size > size_val:
                errors.append(
                    f"Signal '{signal.name}' declared size {size_val}B, but members sum to {total_size}B."
                )

            # Check GPU cache line alignment requirements (must be multiple of 4 or 8)
            if size_val % 4 != 0:
                errors.append(
                    f"Signal '{signal.name}' size {size_val}B is not aligned to GPU word boundary (4B/8B)."
                )

        for node in self.program.nodes:
            # Check manifested size alignment to 64 bytes VRAM cache line constraint
            m_size = 0
            for m in node.manifested.members:
                m_size += TYPE_SIZES.get(m.type_spec, 4)
            m_size += 4  # state_bitmask (4B)

            if m_size > 64:
                errors.append(
                    f"Node '{node.name}' manifested struct size ({m_size}B) exceeds 64B VRAM cache line boundary."
                )

        return errors

    def check_circular_causality(self) -> List[str]:
        """
        Checks for circular dependency loops in causal rules (A -> B -> A).
        """
        graph: Dict[str, Set[str]] = {}
        for rule in self.program.rules:
            src = rule.trigger_type
            dst = rule.target_type
            if src not in graph:
                graph[src] = set()
            graph[src].add(dst)

        cycles = []
        visited = set()
        rec_stack = set()

        def dfs(node: str, path: List[str]):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)

            for neighbor in graph.get(node, []):
                if neighbor not in visited:
                    dfs(neighbor, path)
                elif neighbor in rec_stack:
                    cycle_start = path.index(neighbor)
                    cycle_path = " -> ".join(path[cycle_start:] + [neighbor])
                    cycles.append(f"Circular causality detected: {cycle_path}")

            rec_stack.remove(node)
            path.pop()

        for node in list(graph.keys()):
            if node not in visited:
                dfs(node, [])

        return cycles

    def analyze(self) -> List[str]:
        errors = self.validate_alignment_and_packing()
        errors.extend(self.check_circular_causality())
        return errors
