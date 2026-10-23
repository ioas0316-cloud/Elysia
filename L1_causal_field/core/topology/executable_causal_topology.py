"""
Elysia Executable Causal Topology Engine
========================================
Implements the 5 Core Structural Principles of Executable Causal Topology:
1. DAG-based Dataflow Topology with Topological Sorting & Dirty Flag Propagation
2. Structural Causal Model (SCM) & do-operator (Graph Surgery / Sandbox Snapshot)
3. Structure of Arrays (SoA) Memory Layout & Vectorized SIMD/Batch Execution Engine
4. Hierarchical Encapsulation (Compound Nodes & Component Pattern)
5. Declarative Serialization Schema (JSON) & Target Code Generator (C++, Python, Reports)
"""

from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Union, Callable, Any
import json
import numpy as np


class NodeType(Enum):
    VALUE = "VALUE"             # Base variable / constant / input
    COMPUTE = "COMPUTE"         # Analytical formula / operator
    BRANCH = "BRANCH"           # Conditional logic / trigger / probability gate
    DISTRIBUTION = "DISTRIBUTION" # Stochastic sampling / Monte Carlo output node


class OpCode(Enum):
    INPUT_VAR = "INPUT_VAR"
    CONSTANT = "CONSTANT"
    ADD = "ADD"
    SUBTRACT = "SUBTRACT"
    MULTIPLY = "MULTIPLY"
    DIVIDE = "DIVIDE"
    CLAMP_MIN = "CLAMP_MIN"
    CLAMP_MAX = "CLAMP_MAX"
    PROBABILITY_GATE = "PROBABILITY_GATE"
    NORMAL_DISTRIBUTION = "NORMAL_DISTRIBUTION"
    CUSTOM = "CUSTOM"


@dataclass
class CausalComponent:
    """Component Pattern: Attachable behavior mechanism to DAG Nodes."""
    name: str
    component_type: str
    attributes: Dict[str, Any] = field(default_factory=dict)

    def process(self, value: float) -> float:
        """Default pass-through or modifier logic."""
        if "multiplier" in self.attributes:
            value *= self.attributes["multiplier"]
        if "adder" in self.attributes:
            value += self.attributes["adder"]
        return value


@dataclass
class ExecutableDAGNode:
    """
    DAG Node with explicit dependencies, dirty flags, and structural metadata.
    """
    id: str
    node_type: NodeType = NodeType.VALUE
    op: OpCode = OpCode.CONSTANT
    default_value: float = 0.0
    input_ids: List[str] = field(default_factory=list)
    output_ids: List[str] = field(default_factory=list)

    dirty: bool = True
    cached_value: Optional[float] = None
    components: List[CausalComponent] = field(default_factory=list)
    custom_func: Optional[Callable[..., float]] = None
    formula_str: str = ""

    def add_component(self, component: CausalComponent):
        self.components.append(component)
        self.dirty = True

    def mark_dirty(self, all_nodes: Dict[str, 'ExecutableDAGNode']):
        """Reactive Dirty Flag propagation down the DAG hierarchy."""
        if not self.dirty:
            self.dirty = True
            for out_id in self.output_ids:
                if out_id in all_nodes:
                    all_nodes[out_id].mark_dirty(all_nodes)


class CompoundNode(ExecutableDAGNode):
    """
    Hierarchical Encapsulation: A compound node packaging an entire sub-DAG graph.
    Exposes explicit input_pins and output_pins to the outer graph.
    """
    def __init__(
        self,
        node_id: str,
        sub_nodes: List[ExecutableDAGNode],
        input_pins: Dict[str, str],   # outer_input_name -> inner_node_id
        output_pins: Dict[str, str]   # outer_output_name -> inner_node_id
    ):
        super().__init__(id=node_id, node_type=NodeType.COMPUTE, op=OpCode.CUSTOM)
        self.sub_nodes: Dict[str, ExecutableDAGNode] = {node.id: node for node in sub_nodes}
        self.input_pins = input_pins
        self.output_pins = output_pins

    def evaluate_subgraph(self, input_values: Dict[str, float]) -> Dict[str, float]:
        """Evaluates inner sub-DAG with supplied input pin values."""
        # Map inputs to inner nodes
        for pin_name, inner_id in self.input_pins.items():
            if pin_name in input_values and inner_id in self.sub_nodes:
                self.sub_nodes[inner_id].default_value = input_values[pin_name]
                self.sub_nodes[inner_id].dirty = True

        # Topological evaluation of inner sub-DAG
        sorted_inner = CausalCompiler.topological_sort(list(self.sub_nodes.values()))
        eval_env: Dict[str, float] = {}

        for node in sorted_inner:
            if node.op == OpCode.CONSTANT or node.op == OpCode.INPUT_VAR:
                val = node.default_value
            elif node.op == OpCode.ADD:
                val = sum(eval_env.get(inp, 0.0) for inp in node.input_ids)
            elif node.op == OpCode.MULTIPLY:
                val = 1.0
                for inp in node.input_ids:
                    val *= eval_env.get(inp, 1.0)
            else:
                val = node.default_value

            for comp in node.components:
                val = comp.process(val)

            eval_env[node.id] = val

        # Collect output pin values
        outputs = {}
        for pin_name, inner_id in self.output_pins.items():
            outputs[pin_name] = eval_env.get(inner_id, 0.0)
        return outputs


class StructuralCausalModel:
    """
    SCM (Structural Causal Model) supporting Structural Equations and do-operator Graph Surgery.
    Equations: X_i = f_i(Parents(X_i), U_i)
    """
    def __init__(self, name: str = "SCM"):
        self.name = name
        self.nodes: Dict[str, ExecutableDAGNode] = {}

    def add_node(self, node: ExecutableDAGNode):
        self.nodes[node.id] = node
        # Ensure output relationships
        for inp_id in node.input_ids:
            if inp_id in self.nodes and node.id not in self.nodes[inp_id].output_ids:
                self.nodes[inp_id].output_ids.append(node.id)

    def do_intervention(self, interventions: Dict[str, float]) -> 'StructuralCausalModel':
        """
        do-operator (Graph Surgery):
        Creates a sandbox snapshot where specified target nodes are forced to fixed values,
        cutting their incoming causal links (Parents(X_i) -> empty) without corrupting the original SCM.
        """
        snapshot = StructuralCausalModel(name=f"{self.name}_do_{list(interventions.keys())}")

        # Deep copy DAG topology
        for nid, node in self.nodes.items():
            new_node = ExecutableDAGNode(
                id=node.id,
                node_type=node.node_type,
                op=node.op,
                default_value=node.default_value,
                input_ids=list(node.input_ids),
                output_ids=list(node.output_ids),
                dirty=True,
                formula_str=node.formula_str,
                custom_func=node.custom_func,
                components=[CausalComponent(c.name, c.component_type, dict(c.attributes)) for c in node.components]
            )
            snapshot.nodes[nid] = new_node

        # Perform Graph Surgery for intervened nodes
        for target_id, fix_val in interventions.items():
            if target_id in snapshot.nodes:
                target_node = snapshot.nodes[target_id]
                # Sever incoming edges
                old_parents = list(target_node.input_ids)
                target_node.input_ids = []
                target_node.op = OpCode.CONSTANT
                target_node.default_value = fix_val
                target_node.dirty = True

                # Remove target_id from parents' output_ids in snapshot
                for parent_id in old_parents:
                    if parent_id in snapshot.nodes:
                        if target_id in snapshot.nodes[parent_id].output_ids:
                            snapshot.nodes[parent_id].output_ids.remove(target_id)

        return snapshot


@dataclass
class SoAProgram:
    """
    Flattened Structure of Arrays (SoA) execution program optimized for SIMD / vectorized processing.
    """
    num_nodes: int = 0
    opcodes: List[OpCode] = field(default_factory=list)
    arg_a: List[int] = field(default_factory=list)
    arg_b: List[int] = field(default_factory=list)
    constants: List[float] = field(default_factory=list)
    node_id_map: Dict[str, int] = field(default_factory=dict)     # orig_id -> flat_idx
    flat_to_id_map: Dict[int, str] = field(default_factory=dict)   # flat_idx -> orig_id


class CausalCompiler:
    """
    Compiler that takes DAG Nodes, performs Topological Sorting (Kahn's Algorithm),
    and flattens the logical graph into an SoAProgram execution layout.
    """
    @staticmethod
    def topological_sort(nodes: List[ExecutableDAGNode]) -> List[ExecutableDAGNode]:
        node_map = {n.id: n for n in nodes}
        in_degrees = {n.id: len(n.input_ids) for n in nodes}

        # Dynamically build child outputs map from parent input_ids
        children_map: Dict[str, Set[str]] = {n.id: set(n.output_ids) for n in nodes}
        for n in nodes:
            for inp_id in n.input_ids:
                if inp_id in children_map:
                    children_map[inp_id].add(n.id)

        # Queue with in-degree 0
        queue = [n.id for n in nodes if in_degrees[n.id] == 0]
        sorted_nodes = []

        while queue:
            curr_id = queue.pop(0)
            curr_node = node_map[curr_id]
            sorted_nodes.append(curr_node)

            for out_id in children_map[curr_id]:
                if out_id in in_degrees:
                    in_degrees[out_id] -= 1
                    if in_degrees[out_id] == 0:
                        queue.append(out_id)

        if len(sorted_nodes) != len(nodes):
            raise ValueError("Cycle detected in Causal DAG! Topology must be acyclic.")

        return sorted_nodes

    @staticmethod
    def compile(scm: StructuralCausalModel) -> SoAProgram:
        raw_nodes = list(scm.nodes.values())
        sorted_nodes = CausalCompiler.topological_sort(raw_nodes)

        program = SoAProgram()
        program.num_nodes = len(sorted_nodes)
        program.opcodes = [n.op for n in sorted_nodes]
        program.constants = [n.default_value for n in sorted_nodes]
        program.arg_a = [-1] * program.num_nodes
        program.arg_b = [-1] * program.num_nodes

        for flat_idx, node in enumerate(sorted_nodes):
            program.node_id_map[node.id] = flat_idx
            program.flat_to_id_map[flat_idx] = node.id

        # Map parent input IDs to flattened indices
        for flat_idx, node in enumerate(sorted_nodes):
            if len(node.input_ids) >= 1 and node.input_ids[0] in program.node_id_map:
                program.arg_a[flat_idx] = program.node_id_map[node.input_ids[0]]
            if len(node.input_ids) >= 2 and node.input_ids[1] in program.node_id_map:
                program.arg_b[flat_idx] = program.node_id_map[node.input_ids[1]]

        return program


class SoACausalEvaluator:
    """
    SoA Vectorized Execution Engine (SIMD / Parallel Simulator).
    Evaluates thousands to millions of Monte Carlo simulation samples across flattened contiguous memory.
    """
    def __init__(self, program: SoAProgram, batch_size: int = 10000):
        self.program = program
        self.batch_size = batch_size
        # Flat Memory Layout: shape [num_nodes, batch_size]
        self.memory_pool = np.zeros((self.program.num_nodes, self.batch_size), dtype=np.float32)

    def set_input_batch(self, flat_node_idx: int, data: Union[np.ndarray, List[float], float]):
        """Inject batch data or constant vector into a specific node's memory row."""
        if isinstance(data, (float, int)):
            self.memory_pool[flat_node_idx, :] = float(data)
        else:
            arr = np.asarray(data, dtype=np.float32)
            if arr.shape[0] != self.batch_size:
                raise ValueError(f"Data length {arr.shape[0]} does not match batch_size {self.batch_size}")
            self.memory_pool[flat_node_idx, :] = arr

    def set_input_by_id(self, node_id: str, data: Union[np.ndarray, List[float], float]):
        flat_idx = self.program.node_id_map[node_id]
        self.set_input_batch(flat_idx, data)

    def execute_vectorized_batch(self):
        """Vectorized SIMD batch execution loop following topological SoA order."""
        for node_idx in range(self.program.num_nodes):
            op = self.program.opcodes[node_idx]
            arg_a_idx = self.program.arg_a[node_idx]
            arg_b_idx = self.program.arg_b[node_idx]
            const_val = self.program.constants[node_idx]

            if op == OpCode.CONSTANT:
                self.memory_pool[node_idx, :] = const_val

            elif op == OpCode.INPUT_VAR:
                # Retains injected batch data or defaults to const_val
                if np.all(self.memory_pool[node_idx, :] == 0.0) and const_val != 0.0:
                    self.memory_pool[node_idx, :] = const_val

            elif op == OpCode.ADD:
                in1 = self.memory_pool[arg_a_idx, :] if arg_a_idx >= 0 else 0.0
                in2 = self.memory_pool[arg_b_idx, :] if arg_b_idx >= 0 else 0.0
                self.memory_pool[node_idx, :] = in1 + in2

            elif op == OpCode.SUBTRACT:
                in1 = self.memory_pool[arg_a_idx, :] if arg_a_idx >= 0 else 0.0
                in2 = self.memory_pool[arg_b_idx, :] if arg_b_idx >= 0 else 0.0
                self.memory_pool[node_idx, :] = in1 - in2

            elif op == OpCode.MULTIPLY:
                in1 = self.memory_pool[arg_a_idx, :] if arg_a_idx >= 0 else 1.0
                in2 = self.memory_pool[arg_b_idx, :] if arg_b_idx >= 0 else 1.0
                self.memory_pool[node_idx, :] = in1 * in2

            elif op == OpCode.DIVIDE:
                in1 = self.memory_pool[arg_a_idx, :] if arg_a_idx >= 0 else 0.0
                in2 = self.memory_pool[arg_b_idx, :] if arg_b_idx >= 0 else 1.0
                self.memory_pool[node_idx, :] = in1 / np.maximum(in2, 1e-8)

            elif op == OpCode.CLAMP_MIN:
                in1 = self.memory_pool[arg_a_idx, :] if arg_a_idx >= 0 else const_val
                self.memory_pool[node_idx, :] = np.maximum(in1, const_val)

            elif op == OpCode.CLAMP_MAX:
                in1 = self.memory_pool[arg_a_idx, :] if arg_a_idx >= 0 else const_val
                self.memory_pool[node_idx, :] = np.minimum(in1, const_val)

            elif op == OpCode.PROBABILITY_GATE:
                prob = const_val if arg_a_idx == -1 else self.memory_pool[arg_a_idx, :]
                rolls = np.random.uniform(0.0, 1.0, size=self.batch_size)
                self.memory_pool[node_idx, :] = (rolls < prob).astype(np.float32)

            elif op == OpCode.NORMAL_DISTRIBUTION:
                # const_val = mean, arg_a = std if valid
                std_val = 1.0 if arg_a_idx == -1 else float(np.mean(self.memory_pool[arg_a_idx, :]))
                self.memory_pool[node_idx, :] = np.random.normal(const_val, std_val, size=self.batch_size)

    def get_node_result(self, node_id_or_flat_idx: Union[str, int]) -> np.ndarray:
        if isinstance(node_id_or_flat_idx, str):
            flat_idx = self.program.node_id_map[node_id_or_flat_idx]
        else:
            flat_idx = node_id_or_flat_idx
        return self.memory_pool[flat_idx, :]


class CausalSerializer:
    """Declarative JSON Serialization for Executable Causal Topology."""
    @staticmethod
    def serialize_scm(scm: StructuralCausalModel) -> str:
        data = {
            "name": scm.name,
            "nodes": []
        }
        for node in scm.nodes.values():
            n_data = {
                "id": node.id,
                "node_type": node.node_type.value,
                "op": node.op.value,
                "default_value": node.default_value,
                "input_ids": node.input_ids,
                "output_ids": node.output_ids,
                "formula_str": node.formula_str,
                "components": [
                    {
                        "name": c.name,
                        "component_type": c.component_type,
                        "attributes": c.attributes
                    }
                    for c in node.components
                ]
            }
            data["nodes"].append(n_data)
        return json.dumps(data, indent=2)

    @staticmethod
    def deserialize_scm(json_str: str) -> StructuralCausalModel:
        data = json.loads(json_str)
        scm = StructuralCausalModel(name=data.get("name", "DeserializedSCM"))
        for n_data in data.get("nodes", []):
            node = ExecutableDAGNode(
                id=n_data["id"],
                node_type=NodeType(n_data["node_type"]),
                op=OpCode(n_data["op"]),
                default_value=n_data.get("default_value", 0.0),
                input_ids=n_data.get("input_ids", []),
                output_ids=n_data.get("output_ids", []),
                formula_str=n_data.get("formula_str", "")
            )
            for c_data in n_data.get("components", []):
                comp = CausalComponent(
                    name=c_data["name"],
                    component_type=c_data["component_type"],
                    attributes=c_data.get("attributes", {})
                )
                node.add_component(comp)
            scm.add_node(node)
        return scm


class CausalCodeGenerator:
    """Code Generator Pipeline for C++ headers, Python scripts, and Reports."""
    @staticmethod
    def generate_cpp_header(scm: StructuralCausalModel) -> str:
        program = CausalCompiler.compile(scm)
        cpp_code = f"// Generated C++ SoA Executable Causal Topology Header\n"
        cpp_code += f"#pragma once\n#include <vector>\n#include <algorithm>\n\n"
        cpp_code += f"namespace CausalGen {{\n"
        cpp_code += f"struct {scm.name}SoAProgram {{\n"
        cpp_code += f"    static constexpr size_t NUM_NODES = {program.num_nodes};\n\n"

        cpp_code += f"    // Node ID Mappings\n"
        for nid, flat_i in program.node_id_map.items():
            cpp_code += f"    static constexpr int NODE_{nid.upper()} = {flat_i};\n"

        cpp_code += f"\n    static void execute_batch(float* memory_pool, size_t batch_size) {{\n"

        sorted_nodes = CausalCompiler.topological_sort(list(scm.nodes.values()))
        for flat_i, node in enumerate(sorted_nodes):
            cpp_code += f"        // Node {node.id} ({node.op.value})\n"
            cpp_code += f"        float* out_ptr = &memory_pool[{flat_i} * batch_size];\n"

            if node.op == OpCode.CONSTANT or node.op == OpCode.INPUT_VAR:
                cpp_code += f"        std::fill(out_ptr, out_ptr + batch_size, {node.default_value}f);\n"
            elif node.op == OpCode.ADD:
                a_idx = program.arg_a[flat_i]
                b_idx = program.arg_b[flat_i]
                cpp_code += f"        const float* in1 = &memory_pool[{a_idx} * batch_size];\n"
                cpp_code += f"        const float* in2 = &memory_pool[{b_idx} * batch_size];\n"
                cpp_code += f"        for(size_t i=0; i<batch_size; ++i) out_ptr[i] = in1[i] + in2[i];\n"
            elif node.op == OpCode.MULTIPLY:
                a_idx = program.arg_a[flat_i]
                b_idx = program.arg_b[flat_i]
                cpp_code += f"        const float* in1 = &memory_pool[{a_idx} * batch_size];\n"
                cpp_code += f"        const float* in2 = &memory_pool[{b_idx} * batch_size];\n"
                cpp_code += f"        for(size_t i=0; i<batch_size; ++i) out_ptr[i] = in1[i] * in2[i];\n"
            elif node.op == OpCode.CLAMP_MIN:
                a_idx = program.arg_a[flat_i]
                cpp_code += f"        const float* in1 = &memory_pool[{a_idx} * batch_size];\n"
                cpp_code += f"        for(size_t i=0; i<batch_size; ++i) out_ptr[i] = std::max(in1[i], {node.default_value}f);\n"

        cpp_code += f"    }}\n"
        cpp_code += f"}};\n}}\n"
        return cpp_code

    @staticmethod
    def generate_python_module(scm: StructuralCausalModel) -> str:
        program = CausalCompiler.compile(scm)
        py_code = f"# Generated Python Executable Causal Topology Module\n"
        py_code += f"import numpy as np\n\n"
        py_code += f"class {scm.name}Evaluator:\n"
        py_code += f"    def __init__(self, batch_size=10000):\n"
        py_code += f"        self.batch_size = batch_size\n"
        py_code += f"        self.num_nodes = {program.num_nodes}\n"
        py_code += f"        self.memory_pool = np.zeros(({program.num_nodes}, batch_size), dtype=np.float32)\n"
        py_code += f"        self.node_map = {program.node_id_map}\n\n"

        py_code += f"    def run(self):\n"
        sorted_nodes = CausalCompiler.topological_sort(list(scm.nodes.values()))
        for flat_i, node in enumerate(sorted_nodes):
            py_code += f"        # Node {node.id} ({node.op.value})\n"
            if node.op == OpCode.CONSTANT:
                py_code += f"        self.memory_pool[{flat_i}, :] = {node.default_value}\n"
            elif node.op == OpCode.ADD:
                a_idx = program.arg_a[flat_i]
                b_idx = program.arg_b[flat_i]
                py_code += f"        self.memory_pool[{flat_i}, :] = self.memory_pool[{a_idx}, :] + self.memory_pool[{b_idx}, :]\n"
            elif node.op == OpCode.MULTIPLY:
                a_idx = program.arg_a[flat_i]
                b_idx = program.arg_b[flat_i]
                py_code += f"        self.memory_pool[{flat_i}, :] = self.memory_pool[{a_idx}, :] * self.memory_pool[{b_idx}, :]\n"

        py_code += f"        return self.memory_pool\n"
        return py_code

    @staticmethod
    def generate_causal_report(scm: StructuralCausalModel) -> str:
        report = f"# Causal Flow Report: {scm.name}\n\n"
        report += f"Total Nodes: {len(scm.nodes)}\n\n"
        report += f"## Node Topology:\n"
        for nid, node in scm.nodes.items():
            report += f"- **{nid}** [{node.node_type.value} | {node.op.value}]\n"
            report += f"  - Parents: {node.input_ids}\n"
            report += f"  - Children: {node.output_ids}\n"
            if node.formula_str:
                report += f"  - Formula: `{node.formula_str}`\n"
        return report
