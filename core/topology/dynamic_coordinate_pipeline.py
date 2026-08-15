"""
Causal Topology Engine Python Interface and Core Pipeline.
Provides a Python wrapper around C++/Python data structures for Causal Trace Pool and Dynamic Coordinate Relativization Engine.
"""

from typing import List, Optional, Dict, Any
import math

class PyCausalTracePool:
    """
    Python implementation / wrapper of SoA Causal Trace Pool
    """
    def __init__(self, capacity: int = 0):
        self.count = capacity
        self.values: List[float] = [0.0] * capacity
        self.deltas: List[float] = [0.0] * capacity
        self.operator_ids: List[int] = [0] * capacity
        self.context_masks: List[int] = [0] * capacity
        self.invariance_scores: List[float] = [0.0] * capacity
        self.parent_trace_idx: List[int] = [-1] * capacity
        self.resistor_x: List[float] = [1.0] * capacity
        self.is_axis: List[int] = [0] * capacity

    def resize(self, new_size: int):
        self.count = new_size
        self.values = [0.0] * new_size
        self.deltas = [0.0] * new_size
        self.operator_ids = [0] * new_size
        self.context_masks = [0] * new_size
        self.invariance_scores = [0.0] * new_size
        self.parent_trace_idx = [-1] * new_size
        self.resistor_x = [1.0] * new_size
        self.is_axis = [0] * new_size

    def push_back(self, val: float, delta: float, op_id: int, ctx_mask: int,
                  inv_score: float, parent_idx: int, res_x: float = 1.0, is_axis: int = 0) -> int:
        self.values.append(val)
        self.deltas.append(delta)
        self.operator_ids.append(op_id)
        self.context_masks.append(ctx_mask)
        self.invariance_scores.append(inv_score)
        self.parent_trace_idx.append(parent_idx)
        self.resistor_x.append(res_x)
        self.is_axis.append(is_axis)
        self.count = len(self.values)
        return self.count - 1

    def trace_back(self, start_idx: int) -> List[int]:
        chain = []
        curr = start_idx
        while 0 <= curr < self.count:
            chain.append(curr)
            curr = self.parent_trace_idx[curr]
        return chain


class PyDynamicCoordinateEngine:
    """
    Dynamic Coordinate System & Relativization Engine in Python
    Controls forward signal propagation across variable impedance dials (resistor_x)
    and reflective metaprocess transition (Condensation: x -> Axis, Relativization: Axis -> x).
    """
    def __init__(self, condensation_threshold: float = 0.8, relativization_friction: float = 0.5,
                 min_resistor_x: float = 0.001, max_resistor_x: float = 100.0, dt: float = 0.01, learning_rate: float = 0.1):
        self.condensation_threshold = condensation_threshold
        self.relativization_friction = relativization_friction
        self.min_resistor_x = min_resistor_x
        self.max_resistor_x = max_resistor_x
        self.dt = dt
        self.learning_rate = learning_rate

    def forward_step(self, pool: PyCausalTracePool, inputs: List[float]):
        n = min(pool.count, len(inputs))
        for i in range(n):
            eff_res = self.min_resistor_x if pool.is_axis[i] else max(pool.resistor_x[i], self.min_resistor_x)
            effective_force = inputs[i] / eff_res
            pool.deltas[i] = effective_force * self.dt
            pool.values[i] += pool.deltas[i]

    def capture_trace(self, pool: PyCausalTracePool, node_idx: int, op_id: int, ctx_mask: int, parent_idx: int) -> int:
        if not (0 <= node_idx < pool.count):
            raise IndexError("node_idx out of range in capture_trace")
        return pool.push_back(
            val=pool.values[node_idx],
            delta=pool.deltas[node_idx],
            op_id=op_id,
            ctx_mask=ctx_mask,
            inv_score=pool.invariance_scores[node_idx],
            parent_idx=parent_idx,
            res_x=pool.resistor_x[node_idx],
            is_axis=pool.is_axis[node_idx]
        )

    def reflect_and_mutate(self, pool: PyCausalTracePool, prediction_errors: List[float]):
        n = min(pool.count, len(prediction_errors))
        for i in range(n):
            err = abs(prediction_errors[i])
            if pool.is_axis[i]:
                # Relativization Check: Axis (1,2) -> Variable x
                if err > self.relativization_friction:
                    pool.is_axis[i] = 0
                    pool.resistor_x[i] = min(self.max_resistor_x, pool.resistor_x[i] + self.learning_rate * err + 0.5)
                    pool.invariance_scores[i] = max(0.0, pool.invariance_scores[i] - 0.2)
                else:
                    pool.invariance_scores[i] = min(1.0, pool.invariance_scores[i] + 0.05)
            else:
                # Variable x adaptation & Condensation Check: Variable x -> Axis
                if err < 0.1:
                    pool.invariance_scores[i] = min(1.0, pool.invariance_scores[i] + 0.05 + self.learning_rate * (0.1 - err))
                    pool.resistor_x[i] = max(self.min_resistor_x, pool.resistor_x[i] * 0.9)
                else:
                    pool.resistor_x[i] = min(self.max_resistor_x, pool.resistor_x[i] + self.learning_rate * err)
                    pool.invariance_scores[i] = max(0.0, pool.invariance_scores[i] - self.learning_rate * err)

                if pool.invariance_scores[i] >= self.condensation_threshold:
                    pool.is_axis[i] = 1
                    pool.resistor_x[i] = self.min_resistor_x
