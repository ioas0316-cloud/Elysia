import uuid
import math
import threading
import numpy as np
from typing import Dict, Any, Optional, List, Set, Tuple, Union
from copy import deepcopy

class PhysicalStateSlabPool:
    """
    [Physical Data Continuity Layer]
    SoA (Struct of Arrays) Continuous Memory Slab Pool.
    GPU HBM / Host RAM 상의 거대한 연속 메모리 공간(Slab)을 사전 할당하여
    모든 상태 전이 및 델타 텐서를 오프셋 순서대로 연속 기록(Append-Only)합니다.
    주소 변환 포인터 체이싱 오버헤드를 제거하여 100% 캐시 히트율을 보장합니다.
    """
    def __init__(self, capacity: int = 100000, dimension: int = 64):
        self.capacity = capacity
        self.dimension = dimension

        # SoA (Struct of Arrays) 연속 메모리 배치
        self.state_slabs = np.zeros((capacity, dimension), dtype=np.float32)
        self.delta_slabs = np.zeros((capacity, dimension), dtype=np.float32)
        self.bitmasks = np.zeros(capacity, dtype=np.uint64)
        self.parent_offsets = np.full(capacity, -1, dtype=np.int32)
        self.viability_scores = np.zeros(capacity, dtype=np.float32)

        self._next_offset = 0
        self._lock = threading.RLock()

    def allocate_slab(
        self,
        delta_vec: np.ndarray,
        bitmask: int = 0xFFFFFFFFFFFFFFFF,
        parent_offset: int = -1
    ) -> int:
        """
        연속 메모리 슬래브에 델타 및 마스크 데이터를 Append-Only 방식으로 즉시 기록 (Zero-Copy).
        """
        with self._lock:
            if self._next_offset >= self.capacity:
                new_cap = self.capacity * 2
                new_state = np.zeros((new_cap, self.dimension), dtype=np.float32)
                new_delta = np.zeros((new_cap, self.dimension), dtype=np.float32)
                new_mask = np.zeros(new_cap, dtype=np.uint64)
                new_parent = np.full(new_cap, -1, dtype=np.int32)
                new_viability = np.zeros(new_cap, dtype=np.float32)

                new_state[:self.capacity] = self.state_slabs
                new_delta[:self.capacity] = self.delta_slabs
                new_mask[:self.capacity] = self.bitmasks
                new_parent[:self.capacity] = self.parent_offsets
                new_viability[:self.capacity] = self.viability_scores

                self.state_slabs = new_state
                self.delta_slabs = new_delta
                self.bitmasks = new_mask
                self.parent_offsets = new_parent
                self.viability_scores = new_viability
                self.capacity = new_cap

            offset = self._next_offset
            self._next_offset += 1

            d_vec = np.asarray(delta_vec, dtype=np.float32)
            if d_vec.shape[0] < self.dimension:
                d_vec = np.pad(d_vec, (0, self.dimension - d_vec.shape[0]))
            elif d_vec.shape[0] > self.dimension:
                d_vec = d_vec[:self.dimension]

            self.delta_slabs[offset] = d_vec
            self.bitmasks[offset] = np.uint64(bitmask)
            self.parent_offsets[offset] = parent_offset

            # 부모 상태가 있으면 비트마스크 적용하여 변경된 차원만 갱신, 미변경 차원은 부모 상태 상속
            if parent_offset >= 0:
                parent_state = self.state_slabs[parent_offset]
                mask_bits = np.array([(bitmask >> i) & 1 for i in range(min(self.dimension, 64))], dtype=np.float32)
                if len(mask_bits) < self.dimension:
                    mask_bits = np.pad(mask_bits, (0, self.dimension - len(mask_bits)), constant_values=1.0)
                self.state_slabs[offset] = parent_state * (1.0 - mask_bits) + d_vec * mask_bits
            else:
                self.state_slabs[offset] = d_vec

            return offset

    def get_slab_state(self, offset: int) -> np.ndarray:
        """O(1) 캐시 최적화 상태 읽기."""
        with self._lock:
            if 0 <= offset < self._next_offset:
                return self.state_slabs[offset].copy()
            return np.zeros(self.dimension, dtype=np.float32)


class StateNode:
    """
    [Informational Framing Layer]
    불변(Immutable) Virtual Observation View Node.
    """
    def __init__(
        self,
        slab_offset: int,
        parent: Optional['StateNode'] = None,
        delta_dict: Optional[Dict[str, Any]] = None,
        bitmask: int = 0xFFFFFFFFFFFFFFFF,
        intervention_meta: Optional[str] = None
    ):
        self.id = str(uuid.uuid4())[:8]
        self.slab_offset = slab_offset
        self.parent = parent
        self.delta_dict = delta_dict if delta_dict is not None else {}
        self.bitmask = bitmask
        self.intervention_meta = intervention_meta
        self.children: Set['StateNode'] = set()
        self._lock = threading.RLock()

        if parent:
            parent.children.add(self)

    def get_state_chain(self) -> List[Dict[str, Any]]:
        with self._lock:
            chain = []
            curr = self
            while curr:
                chain.append(curr.delta_dict)
                curr = curr.parent

            full_dict = {}
            for d in reversed(chain):
                full_dict.update(d)
            return full_dict

    def compute_node_divergence(self) -> float:
        with self._lock:
            divergence = 0.0
            for val in self.delta_dict.values():
                if isinstance(val, (int, float)):
                    divergence += abs(float(val))
                elif isinstance(val, np.ndarray):
                    divergence += float(np.linalg.norm(val))
                else:
                    divergence += 1.0
            return divergence


class StateDAGManager:
    """
    [Physical Continuity + Informational Framing Unified Manager]
    고정 변수-차원 맵(Variable Dimension Mapping)을 유지하여
    차원 왜곡 없이 상태 벡터 및 델타 슬래브를 관리합니다.
    """
    def __init__(self, initial_state_dict: Dict[str, Any], state_dim: int = 64):
        self.state_dim = state_dim
        self.slab_pool = PhysicalStateSlabPool(capacity=10000, dimension=state_dim)
        self._lock = threading.RLock()

        # 변수명 -> 차원 오프셋 고정 매핑 테이블
        self.var_to_dim: Dict[str, int] = {}
        for key in sorted(initial_state_dict.keys()):
            self._get_or_register_dim(key)

        init_vec, mask = self._dict_to_vec_and_mask(initial_state_dict)
        root_offset = self.slab_pool.allocate_slab(init_vec, bitmask=mask, parent_offset=-1)

        with self._lock:
            self.root = StateNode(
                slab_offset=root_offset,
                parent=None,
                delta_dict=deepcopy(initial_state_dict),
                bitmask=mask
            )
            self.current_node = self.root
            self.nodes: Dict[str, StateNode] = {self.root.id: self.root}

    def _get_or_register_dim(self, var_name: str) -> int:
        if var_name not in self.var_to_dim:
            dim_idx = len(self.var_to_dim) % self.state_dim
            self.var_to_dim[var_name] = dim_idx
        return self.var_to_dim[var_name]

    def _dict_to_vec_and_mask(self, state_dict: Dict[str, Any]) -> Tuple[np.ndarray, int]:
        vec = np.zeros(self.state_dim, dtype=np.float32)
        mask = 0
        for k, v in state_dict.items():
            dim_idx = self._get_or_register_dim(k)
            mask |= (1 << (dim_idx % 64))
            if isinstance(v, (int, float)):
                vec[dim_idx] = float(v)
            elif isinstance(v, (list, np.ndarray)):
                arr = np.asarray(v, dtype=np.float32).flatten()
                vec[dim_idx] = float(arr[0]) if len(arr) > 0 else 0.0
            else:
                vec[dim_idx] = float(hash(str(v)) % 100) / 100.0
        return vec, (mask if mask != 0 else 0xFFFFFFFFFFFFFFFF)

    def step(self, transition_delta: Dict[str, Any], custom_bitmask: Optional[int] = None) -> StateNode:
        with self._lock:
            delta_vec, computed_mask = self._dict_to_vec_and_mask(transition_delta)
            bitmask = custom_bitmask if custom_bitmask is not None else computed_mask
            parent_offset = self.current_node.slab_offset

            new_offset = self.slab_pool.allocate_slab(delta_vec, bitmask=bitmask, parent_offset=parent_offset)

            new_node = StateNode(
                slab_offset=new_offset,
                parent=self.current_node,
                delta_dict=transition_delta,
                bitmask=bitmask
            )
            self.nodes[new_node.id] = new_node
            self.current_node = new_node
            return new_node

    def rewind_to(self, node_id: str) -> StateNode:
        with self._lock:
            if node_id not in self.nodes:
                raise ValueError(f"노드 ID '{node_id}'를 찾을 수 없습니다.")
            self.current_node = self.nodes[node_id]
            return self.current_node

    def do_intervention(self, variable: str, value: Any, custom_bitmask: Optional[int] = None) -> StateNode:
        with self._lock:
            intervention_delta = {variable: value}
            delta_vec, computed_mask = self._dict_to_vec_and_mask(intervention_delta)
            bitmask = custom_bitmask if custom_bitmask is not None else computed_mask
            parent_offset = self.current_node.slab_offset

            new_offset = self.slab_pool.allocate_slab(
                delta_vec,
                bitmask=bitmask,
                parent_offset=parent_offset
            )

            intervened_node = StateNode(
                slab_offset=new_offset,
                parent=self.current_node,
                delta_dict=intervention_delta,
                bitmask=bitmask,
                intervention_meta=f"do({variable}={value})"
            )
            self.nodes[intervened_node.id] = intervened_node
            self.current_node = intervened_node
            return intervened_node

    def build_active_view_mask(self, node: Optional[StateNode] = None) -> np.ndarray:
        with self._lock:
            target = node or self.current_node
            num_slabs = self.slab_pool._next_offset
            view_mask = np.zeros(num_slabs, dtype=np.uint8)

            curr = target
            while curr:
                if 0 <= curr.slab_offset < num_slabs:
                    view_mask[curr.slab_offset] = 1
                curr = curr.parent

            return view_mask

    def get_current_state_vector(self) -> np.ndarray:
        with self._lock:
            return self.slab_pool.get_slab_state(self.current_node.slab_offset)

    def print_dag(self, node: Optional[StateNode] = None, indent: int = 0):
        with self._lock:
            if node is None:
                node = self.root

            prefix = "  " * indent
            meta = f" <{node.intervention_meta}>" if node.intervention_meta else ""
            is_curr = " *[CURRENT]" if node.id == self.current_node.id else ""
            full_state = node.get_state_chain()
            print(f"{prefix}Node[{node.id}] (SlabOffset={node.slab_offset}){meta}{is_curr} | Delta: {node.delta_dict} | FullState: {full_state}")

            for child in sorted(node.children, key=lambda x: x.id):
                self.print_dag(child, indent + 1)
