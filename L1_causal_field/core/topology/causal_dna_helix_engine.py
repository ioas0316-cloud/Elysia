"""
Elysia Core Engine: Causal DNA Helix Engine (수치/위상 차원 변환 및 이중나선 인과 엔진)
========================================================================================
"곱셈과 나눗셈이 정해진 틀 안에서 양을 조절하는 도구라면,
 로그는 폭주하는 비선형 연산의 관계망을 정돈된 선형 공간으로 재배치하는 차원 변환기이다."

본 모듈은 지수적으로 폭주하는 수치 공간을 선형/가법적 로그 스케일 공간으로 압축하고,
연속적 가변 파동 가닥(Variable Wave Strand)과 불변 인과 껍질 가닥(Invariant Shell Strand)으로
구성된 이중나선(Double Helix) 위상 공간 상에서 4대 원자 인과 상태(Phase-Lock, Orthogonality,
Direct Action, Feedback Action)의 크로네커 적(Kronecker Product) 텐서 연산을 수행합니다.

주요 구성 요소를 포함합니다:
1. MultiRotorBufferManager: VRAM (Fast), System RAM (Mid), SSD (Slow) 속도별 로터 계층 관리자.
2. LogarithmicMapper: 지수 폭주 방지 및 Weber-Fechner 감각 동기화.
3. DoubleHelixTopology: 연속 파동축과 불변 껍질축의 상보적 회전 위상 사상.
4. CausalAtomicTensor: 2x2 4대 원자 인과 상태 및 크로네커 적 기반 인과 체인 전개.
5. MultiLODDialIndexer: 옥타브 스케일 다이얼 및 계층적 O(1) 블록 슬라이싱 탐색기.
6. ContextPlaneTransformer: 문맥 면 C 매트릭스 사상 (M' = C · M · C^T).
7. WordPhaseManifold: 초/중/종성 음절 인과 행렬 결합 및 단어 위상 매니폴드.
8. CausalDNAHelixEngine: 전체 파이프라인 통합 엔진.
"""

import os
import math
import torch
import numpy as np
from typing import Dict, List, Tuple, Any, Optional


class MultiRotorBufferManager:
    """
    GTX 1060 3GB VRAM (Fast Rotor / 초점 렌즈), System RAM (Mid Rotor / 작업 기억),
    SSD (Slow Rotor / 백본 지형) 다중 로터 계층 메모리 관리자.
    (FP32 전용, Kronecker Factor 메모리 절약)
    """
    def __init__(self, vram_budget_mb: int = 2000, disk_storage_path: str = "./topology_ssd.bin"):
        self.vram_budget_bytes = vram_budget_mb * 1024 * 1024
        self.current_vram_usage = 0

        self.gpu_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cpu_device = torch.device("cpu")
        self.disk_path = disk_storage_path

        # Rotors
        self.vram_rotor: Dict[str, torch.Tensor] = {}   # Fast Rotor: GPU FP32 Active Tensors
        self.ram_rotor: Dict[str, torch.Tensor] = {}    # Mid Rotor: System RAM Pinned Tensors
        self.focus_scores: Dict[str, float] = {}        # Node Focus Scores (Attention Frequency)

        self._init_ssd_storage()

    def _init_ssd_storage(self) -> None:
        if not os.path.exists(self.disk_path):
            try:
                with open(self.disk_path, "wb") as f:
                    f.seek(1024 * 1024 * 50)  # 50MB 초기 파일 셸 생성
                    f.write(b'\0')
            except Exception:
                pass

    def push_node(self, node_id: str, tensor: torch.Tensor, initial_focus: float = 0.0) -> None:
        """FP32 텐서를 RAM Pinned Memory로 1차 등록 후 계층 동기화"""
        tensor = tensor.to(dtype=torch.float32)
        if self.cpu_device.type == "cpu":
            try:
                pinned_tensor = tensor.to(self.cpu_device).pin_memory()
            except Exception:
                pinned_tensor = tensor.to(self.cpu_device)
        else:
            pinned_tensor = tensor.to(self.cpu_device)

        self.ram_rotor[node_id] = pinned_tensor
        self.focus_scores[node_id] = initial_focus
        self._sync_rotor_hierarchy(node_id)

    def set_focus(self, node_id: str, focus_delta: float) -> None:
        """다이얼 회전에 따른 주의(Attention) 주파수 업데이트"""
        if node_id in self.focus_scores:
            self.focus_scores[node_id] += focus_delta
            self._sync_rotor_hierarchy(node_id)

    def get_active_tensor(self, node_id: str) -> torch.Tensor:
        """연산에 필요한 텐서를 VRAM으로 승격시켜 반환"""
        self.set_focus(node_id, 1.0)
        if node_id in self.vram_rotor:
            return self.vram_rotor[node_id]
        elif node_id in self.ram_rotor:
            return self.ram_rotor[node_id].to(self.gpu_device)
        else:
            raise KeyError(f"Node '{node_id}' not found in buffer manager.")

    def _sync_rotor_hierarchy(self, target_node_id: str) -> None:
        sorted_nodes = sorted(self.focus_scores.keys(), key=lambda k: self.focus_scores[k], reverse=True)

        for node_id in sorted_nodes:
            tensor_size = self._get_tensor_size(node_id)
            if self.current_vram_usage + tensor_size <= self.vram_budget_bytes:
                if node_id not in self.vram_rotor and node_id in self.ram_rotor:
                    self._promote_to_vram(node_id)
            else:
                if node_id in self.vram_rotor and node_id != target_node_id:
                    self._demote_to_ram(node_id)

    def _promote_to_vram(self, node_id: str) -> None:
        if node_id in self.ram_rotor:
            tensor = self.ram_rotor[node_id]
            self.vram_rotor[node_id] = tensor.to(self.gpu_device, non_blocking=True)
            self.current_vram_usage += self._get_tensor_size(node_id)

    def _demote_to_ram(self, node_id: str) -> None:
        if node_id in self.vram_rotor:
            del self.vram_rotor[node_id]
            self.current_vram_usage -= self._get_tensor_size(node_id)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _get_tensor_size(self, node_id: str) -> int:
        if node_id in self.ram_rotor:
            return self.ram_rotor[node_id].element_size() * self.ram_rotor[node_id].nelement()
        return 0


class LogarithmicMapper:
    """
    log(A * B) = log A + log B 변환을 통한 지수 폭주 방지 및 Weber-Fechner 감각 동기화
    """
    def __init__(self, eps: float = 1e-7):
        self.eps = eps

    def to_log_space(self, tensor: torch.Tensor) -> torch.Tensor:
        """곱셈 연산 공간을 가법(Additive) 선형 공간으로 압축"""
        return torch.log(torch.abs(tensor) + self.eps)

    def from_log_space(self, log_tensor: torch.Tensor) -> torch.Tensor:
        """로그 공간에서 원래 선형 스케일로 복원"""
        return torch.exp(log_tensor) - self.eps

    def quantize_octaves(self, log_tensor: torch.Tensor, num_octaves: int = 8) -> torch.Tensor:
        """연속 로그 스케일을 이산적 옥타브(Octave) 마디로 정규화"""
        min_val, max_val = log_tensor.min(), log_tensor.max()
        scale = (max_val - min_val) / num_octaves + self.eps
        octave_indices = torch.floor((log_tensor - min_val) / scale)
        return torch.clamp(octave_indices, 0, num_octaves - 1)


class DoubleHelixTopology:
    """
    연속 파동 가닥(가변축)과 이산 인과 껍질(불변 시그니처 축)의 상보적 회전 위상 매핑
    """
    def __init__(self, dimension: int = 16):
        self.dimension = dimension

    def generate_helix_pair(self, base_signal: torch.Tensor, theta: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        위상 다이얼 회전각(theta)에 따라 변이하는 파동축과 불변 껍질축 생성
        """
        # 1. Invariant Shell Axis (불변 정체성 껍질)
        shell_axis = torch.sign(base_signal) * torch.abs(base_signal).mean()

        # 2. Variable Wave Axis (위상 회전 파동축)
        cos_t = math.cos(theta)
        sin_t = math.sin(theta)

        # 2D 위상 회전 매트릭스
        rot_matrix = torch.tensor([
            [cos_t, -sin_t],
            [sin_t,  cos_t]
        ], dtype=torch.float32, device=base_signal.device)

        # 신호를 2D 짝으로 재구성 후 회전
        original_shape = base_signal.shape
        flat_signal = base_signal.view(-1)
        if len(flat_signal) % 2 != 0:
            flat_signal = torch.cat([flat_signal, torch.zeros(1, device=base_signal.device)])

        reshaped = flat_signal.view(-1, 2)
        rotated_wave = torch.matmul(reshaped, rot_matrix).view(-1)[:base_signal.numel()].view(original_shape)

        return rotated_wave, shell_axis


class CausalAtomicTensor:
    """
    2x2 4대 원자 인과 상태 (Phase-Lock, Orthogonality, Direct Action, Feedback Action) 및
    Kronecker Product 기반 인과 체인 확장기
    """
    def __init__(self, device: torch.device = torch.device('cpu')):
        self.device = device
        # 2x2 원자 인과 상태 기저
        self.S_LOCK     = torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.float32, device=device)  # (1,1) 위상 고정
        self.S_ORTH     = torch.tensor([[0.0, 0.0], [0.0, 0.0]], dtype=torch.float32, device=device)  # (0,0) 직교/단절
        self.S_DIRECT   = torch.tensor([[1.0, 0.0], [0.0, 0.0]], dtype=torch.float32, device=device)  # (1,0) 순방향 인과
        self.S_FEEDBACK = torch.tensor([[0.0, 1.0], [0.0, 0.0]], dtype=torch.float32, device=device)  # (0,1) 역방향 피드백

    def get_atomic_state_by_phase(self, phase_val: float) -> torch.Tensor:
        """사분면 경계(Quarter-Phase Boundary, pi/2) 기반 자발적 상전이"""
        normalized_phase = (phase_val % (2 * math.pi)) / (math.pi / 2)
        quadrant = int(normalized_phase) % 4

        if quadrant == 0:
            return self.S_DIRECT
        elif quadrant == 1:
            return self.S_LOCK
        elif quadrant == 2:
            return self.S_FEEDBACK
        else:
            return self.S_ORTH

    def kronecker_expand(self, tensor_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Kronecker Product (A ⊗ B ⊗ C...)를 통한 복합 인과 매트릭스 텐서 확장
        조건문 없이 GEMM 고속 전개
        """
        result = tensor_list[0]
        for t in tensor_list[1:]:
            result = torch.kron(result, t)
        return result


class MultiLODDialIndexer:
    """
    옥타브 스케일 다이얼을 통한 거시/미시 계층적 O(1) 블록 슬라이싱 탐색기
    """
    def __init__(self, log_mapper: Optional[LogarithmicMapper] = None):
        self.log_mapper = log_mapper or LogarithmicMapper()

    def slice_by_lod(self, causal_matrix: torch.Tensor, lod_level: int) -> torch.Tensor:
        """
        Kronecker 자승 구조의 특성을 활용해 상위 옥타브 블록만 즉시 슬라이싱
        """
        step = 2 ** lod_level
        dim = causal_matrix.shape[0]
        if step >= dim or step <= 1:
            return causal_matrix
        return causal_matrix[0:dim:step, 0:dim:step]


class ContextPlaneTransformer:
    """
    문맥 면(Context Plane) 변환 행렬 C를 생성하고,
    음절/단어 인과 행렬에 사상(Projection)을 수행하는 변환기
    """
    def __init__(self, dim: int = 4, device: torch.device = torch.device('cpu')):
        self.dim = dim
        self.device = device

    def create_context_plane(self, context_type: str, intensity: float = 1.0) -> torch.Tensor:
        """
        특정 문맥(명사형, 동사형, 중립 등)을 나타내는 4x4 변환 행렬 C 생성
        """
        I = torch.eye(self.dim, dtype=torch.float32, device=self.device)

        if context_type == "NOUN_FIELD":
            C = I + intensity * torch.tensor([
                [0.2, 0.1, 0.0, 0.0],
                [0.1, 0.2, 0.0, 0.0],
                [0.0, 0.0, 0.1, 0.0],
                [0.0, 0.0, 0.0, 0.1]
            ], dtype=torch.float32, device=self.device)
        elif context_type == "VERB_FIELD":
            C = torch.tensor([
                [0.0, 1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.5, 0.5],
                [0.0, 0.0, 0.5, 0.5]
            ], dtype=torch.float32, device=self.device) * intensity
        else:
            C = I

        return C / (torch.norm(C) + 1e-7)

    def apply_context(self, causal_matrix: torch.Tensor, context_plane: torch.Tensor) -> torch.Tensor:
        """
        인과 행렬 M에 문맥 면 C를 양방향 사상: M' = C · M · C^T
        """
        return torch.matmul(torch.matmul(context_plane, causal_matrix), context_plane.T)


class WordPhaseManifold:
    """
    초-중-종성 음절 인과 행렬을 결합하고,
    문맥 면을 거쳐 단어 단위 위상 매니폴드로 전개하는 모듈
    """
    def __init__(self, engine: 'CausalDNAHelixEngine'):
        self.engine = engine
        self.device = engine.device
        self.context_transformer = ContextPlaneTransformer(dim=4, device=self.device)

    def build_syllable_matrix(self, cho_theta: float, jung_theta: float, jong_theta: Optional[float] = None) -> torch.Tensor:
        """
        초성, 중성, 종성의 위상각(theta)에 따른 4x4 음절 인과 행렬 합성
        """
        atomic = self.engine.causal_atomic

        M_cho = atomic.get_atomic_state_by_phase(cho_theta)    # 2x2
        M_jung = atomic.get_atomic_state_by_phase(jung_theta)  # 2x2

        # 초성과 중성의 크로네커 결합 (2x2 -> 4x4)
        M_syllables = atomic.kronecker_expand([M_cho, M_jung])

        if jong_theta is not None:
            M_jong = atomic.get_atomic_state_by_phase(jong_theta)
            M_jong_4x4 = atomic.kronecker_expand([M_jong, atomic.S_LOCK])
            M_syllables = torch.matmul(M_syllables, M_jong_4x4)

        return M_syllables

    def construct_word_manifold(self, syllables_thetas: List[Tuple], context_type: str = "NEUTRAL") -> Dict[str, Any]:
        """
        여러 음절을 묶어 단어 단위 위상 매니폴드 생성
        """
        context_plane = self.context_transformer.create_context_plane(context_type)
        contextual_syllables = []

        for item in syllables_thetas:
            cho = item[0]
            jung = item[1]
            jong_t = item[2] if len(item) > 2 else None

            M_s = self.build_syllable_matrix(cho, jung, jong_t)
            M_s_ctx = self.context_transformer.apply_context(M_s, context_plane)
            contextual_syllables.append(M_s_ctx)

        word_manifold = contextual_syllables[0]
        for M_next in contextual_syllables[1:]:
            word_manifold = torch.matmul(word_manifold, M_next)

        return {
            "context_type": context_type,
            "context_plane": context_plane,
            "word_manifold": word_manifold,
            "syllable_matrices": contextual_syllables
        }


class CausalDNAHelixEngine:
    """
    통합 수치/위상 차원 변환 및 이중나선 인과 엔진
    """
    def __init__(self, vram_budget_mb: int = 2000):
        self.buffer_mgr = MultiRotorBufferManager(vram_budget_mb=vram_budget_mb)
        self.device = self.buffer_mgr.gpu_device

        self.log_mapper = LogarithmicMapper()
        self.helix_topo = DoubleHelixTopology(dimension=16)
        self.causal_atomic = CausalAtomicTensor(device=self.device)
        self.lod_indexer = MultiLODDialIndexer(self.log_mapper)

    def process_dial_rotation(
        self,
        node_id: str,
        raw_signal: torch.Tensor,
        dial_theta: float,
        lod_level: int = 0
    ) -> Dict[str, Any]:
        """
        단일 파이프라인 실시간 실행 루틴
        1. 텐서 노드 버퍼 등록 및 주파수 승격
        2. 로그 스케일 압축
        3. 이중나선 위상 회전 (가변축/불변축 분리)
        4. 위상각 기반 4상태 인과 텐서 결정 및 크로네커 확장
        5. Multi-LOD 탐색 및 활성 인과 매트릭스 출력
        """
        self.buffer_mgr.push_node(node_id, raw_signal, initial_focus=1.0)
        active_signal = self.buffer_mgr.get_active_tensor(node_id)

        # 로그 스케일 압축
        log_signal = self.log_mapper.to_log_space(active_signal)

        # 이중나선 위상 회전 (가변축/불변축 분리)
        wave_axis, shell_axis = self.helix_topo.generate_helix_pair(log_signal, dial_theta)

        # 4상태 인과 텐서 결정 및 크로네커 확장
        base_state_1 = self.causal_atomic.get_atomic_state_by_phase(dial_theta)
        base_state_2 = self.causal_atomic.get_atomic_state_by_phase(dial_theta + math.pi / 4)

        expanded_causal_matrix = self.causal_atomic.kronecker_expand([base_state_1, base_state_2])

        # Multi-LOD Indexing
        lod_matrix = self.lod_indexer.slice_by_lod(expanded_causal_matrix, lod_level=lod_level)

        return {
            "node_id": node_id,
            "wave_axis": wave_axis,
            "shell_axis": shell_axis,
            "causal_matrix": lod_matrix,
            "vram_usage_bytes": self.buffer_mgr.current_vram_usage
        }
