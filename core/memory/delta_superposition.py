import threading
import copy
import math
import numpy as np
from typing import Dict, Any, List, Optional, Set, Tuple, Union

class ImmutableBaseSlab:
    """
    [Layer 1] Immutable Base Slab (물리적 기저 상태)
    물리 메모리의 continuous 공간에 단 하나만 존재하며 완전히 불변(Read-Only)을 유지합니다.
    어떤 스레드나 분기에서도 Write를 가하지 않으므로 메모리 락 및 Cache Invalidation 오버헤드가 0입니다.
    """
    def __init__(self, base_data: Union[Dict[str, Any], np.ndarray]):
        if isinstance(base_data, np.ndarray):
            self._is_vector = True
            # Lock writing on numpy array to enforce immutability
            self._vector_data = base_data.copy()
            self._vector_data.flags.writeable = False
            self._dict_data = None
        elif isinstance(base_data, dict):
            self._is_vector = False
            self._dict_data = copy.deepcopy(base_data)
            self._vector_data = None
        else:
            raise TypeError("[ImmutableBaseSlab] base_data must be a dict or a numpy ndarray.")

    @property
    def is_vector(self) -> bool:
        return self._is_vector

    @property
    def vector_data(self) -> Optional[np.ndarray]:
        return self._vector_data

    @property
    def dict_data(self) -> Optional[Dict[str, Any]]:
        return self._dict_data


class LockFreeDeltaRingBuffer:
    """
    [Layer 2] Delta Superposition Ring (변위 중첩 링 버퍼)
    Lock-Free Single-Producer / Multi-Consumer 패러다임을 모방하는 연속 링 버퍼.
    모든 분기의 변화량이 순차 수집되는 Append-Only 변위 버퍼로,
    글로벌 파형 인덱스(Wave Index)를 부여하고 링 버퍼 회전에 의해 Old Delta가 소멸합니다.
    """
    def __init__(self, capacity: int = 1024, vector_dim: Optional[int] = None):
        self.capacity = capacity
        self.vector_dim = vector_dim

        self._atomic_head = 0
        self._lock = threading.Lock()  # High-performance internal synchronization for atomic head increment

        # Ring buffer storage for key-value deltas and vector deltas
        self._kv_buffer: List[Optional[Tuple[str, Any]]] = [None] * capacity
        self._wave_indices: List[int] = [-1] * capacity

        if vector_dim is not None:
            # Pre-allocated contiguous memory buffer for SIMD vector superposition
            self._vector_buffer = np.zeros((capacity, vector_dim), dtype=np.float32)
        else:
            self._vector_buffer = None

    def push_kv_delta(self, key: str, value: Any) -> int:
        """
        Key-Value 변위를 버퍼에 연속 축적하고 글로벌 파형 인덱스를 반환.
        """
        with self._lock:
            wave_idx = self._atomic_head
            self._atomic_head += 1
            slot = wave_idx % self.capacity
            self._kv_buffer[slot] = (key, value)
            self._wave_indices[slot] = wave_idx
            return wave_idx

    def push_vector_delta(self, delta_vector: np.ndarray) -> int:
        """
        SIMD Vector 변위를 contiguous 버퍼에 연속 축적하고 글로벌 파형 인덱스를 반환.
        """
        if self._vector_buffer is None:
            raise ValueError("[LockFreeDeltaRingBuffer] Vector buffer not initialized. Specify vector_dim at init.")

        delta_vec = np.asarray(delta_vector, dtype=np.float32)
        if delta_vec.shape[0] != self.vector_dim:
            raise ValueError(f"[LockFreeDeltaRingBuffer] Vector dimension mismatch: expected {self.vector_dim}, got {delta_vec.shape[0]}")

        with self._lock:
            wave_idx = self._atomic_head
            self._atomic_head += 1
            slot = wave_idx % self.capacity
            self._vector_buffer[slot] = delta_vec
            self._wave_indices[slot] = wave_idx
            return wave_idx

    def get_kv_delta(self, wave_idx: int) -> Optional[Tuple[str, Any]]:
        slot = wave_idx % self.capacity
        if self._wave_indices[slot] == wave_idx:
            return self._kv_buffer[slot]
        return None  # Expired due to ring buffer wrap-around

    def get_vector_delta(self, wave_idx: int) -> Optional[np.ndarray]:
        if self._vector_buffer is None:
            return None
        slot = wave_idx % self.capacity
        if self._wave_indices[slot] == wave_idx:
            return self._vector_buffer[slot]
        return None  # Expired due to ring buffer wrap-around

    def composite_vector_superposition(self, base_vector: np.ndarray, active_wave_indices: List[int]) -> np.ndarray:
        """
        [SIMD / FMA Vector Composite]
        S_effective = S_base + sum(alpha_i * Delta_i)
        비트마스크 / 활성 파형 인덱스를 이용하여 SIMD vectorized FMA 수량화 연산으로 상태를 중첩 합성합니다.
        """
        result = base_vector.copy()
        if self._vector_buffer is None or not active_wave_indices:
            return result

        # Filter active indices that are still valid in the ring buffer
        valid_slots = []
        for wave_idx in active_wave_indices:
            slot = wave_idx % self.capacity
            if self._wave_indices[slot] == wave_idx:
                valid_slots.append(slot)

        if not valid_slots:
            return result

        # Vectorized sum of valid delta vectors (SIMD accelerated via NumPy/BLAS)
        deltas = self._vector_buffer[valid_slots]  # Shape: (K, vector_dim)
        sum_delta = np.sum(deltas, axis=0)
        np.add(result, sum_delta, out=result)
        return result

    @property
    def total_deltas_pushed(self) -> int:
        return self._atomic_head


class ObserverView:
    """
    [Layer 3] Virtual Observer View (가상 관측 레이어)
    물리 데이터의 복사본을 소유하지 않는 가상 관측 프레임 (Virtual View).
    소유 오버헤드는 단 몇 비트의 활성 파형 인덱스/비트마스크뿐입니다 (Zero-Copy Branching).
    """
    def __init__(self, engine: 'DeltaSuperpositionEngine', active_wave_indices: Optional[Union[List[int], Set[int]]] = None):
        self._engine = engine
        if active_wave_indices is None:
            self.active_indices: Set[int] = set()
        elif isinstance(active_wave_indices, set):
            self.active_indices = active_wave_indices
        else:
            self.active_indices = set(active_wave_indices)

    def branch_and_apply_kv(self, key: str, value: Any) -> 'ObserverView':
        """
        [Zero-Copy Key-Value Branching]
        기존 관측 상태를 복사하지 않고, 신규 변위를 링 버퍼에 추가한 뒤 인덱스만 확장된 새 뷰를 반환.
        """
        new_wave_idx = self._engine.ring_buffer.push_kv_delta(key, value)
        new_indices = set(self.active_indices)
        new_indices.add(new_wave_idx)
        return ObserverView(self._engine, new_indices)

    def branch_and_apply_vector(self, delta_vector: np.ndarray) -> 'ObserverView':
        """
        [Zero-Copy Vector Branching]
        기존 관측 상태를 복사하지 않고, 신규 변위 벡터를 링 버퍼에 추가한 뒤 인덱스만 확장된 새 뷰를 반환.
        """
        new_wave_idx = self._engine.ring_buffer.push_vector_delta(delta_vector)
        new_indices = set(self.active_indices)
        new_indices.add(new_wave_idx)
        return ObserverView(self._engine, new_indices)

    def observe(self) -> Union[Dict[str, Any], np.ndarray]:
        """
        [Superposition Composite]
        기저 상태(Base)를 읽고 관측 비트마스크(active_indices)에 활성화된 변위만 중첩 합성.
        """
        base_slab = self._engine.base_slab
        if base_slab.is_vector:
            return self._engine.ring_buffer.composite_vector_superposition(
                base_slab.vector_data, sorted(self.active_indices)
            )
        else:
            # Key-Value superposition
            effective_state = dict(base_slab.dict_data)
            for wave_idx in sorted(self.active_indices):
                kv = self._engine.ring_buffer.get_kv_delta(wave_idx)
                if kv is not None:
                    key, val = kv
                    effective_state[key] = val
            return effective_state


class DeltaSuperpositionEngine:
    """
    불변 기저 상태(Base State)와 변위 중첩(Delta Superposition) 버퍼 기반의
    Zero-Copy 관측 아키텍처 코어 엔진.
    """
    def __init__(self, base_state: Union[Dict[str, Any], np.ndarray], ring_capacity: int = 1024):
        self.base_slab = ImmutableBaseSlab(base_state)
        vector_dim = self.base_slab.vector_data.shape[0] if self.base_slab.is_vector else None
        self.ring_buffer = LockFreeDeltaRingBuffer(capacity=ring_capacity, vector_dim=vector_dim)

    def create_root_view(self) -> ObserverView:
        """
        물리 데이터 복사 없이 텅 빈 관측 비트마스크만 소유하는 루트 가상 관측 뷰 생성.
        """
        return ObserverView(self, active_wave_indices=set())

    def create_view(self, active_wave_indices: Union[List[int], Set[int]]) -> ObserverView:
        """
        지정된 활성 파형 인덱스 집합을 갖는 가상 관측 뷰 생성.
        """
        return ObserverView(self, active_wave_indices=active_wave_indices)
