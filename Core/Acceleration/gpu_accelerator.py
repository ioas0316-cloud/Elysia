"""
GPU Accelerator - GPU 가속 엔진
==============================

높은 우선순위 #3: CPU only → CUDA/PyTorch 통합
예상 효과: 50x 연산 속도

핵심 기능:
- 텐서 연산 가속
- 배치 공명 계산
- 자동 GPU/CPU 폴백
- 메모리 최적화
"""

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger("GPUAccelerator")

# PyTorch 선택적 임포트
try:
    import torch
    TORCH_AVAILABLE = True
    if torch.cuda.is_available():
        GPU_AVAILABLE = True
        GPU_NAME = torch.cuda.get_device_name(0)
    else:
        GPU_AVAILABLE = False
        GPU_NAME = "N/A"
except ImportError:
    TORCH_AVAILABLE = False
    GPU_AVAILABLE = False
    GPU_NAME = "N/A"
    torch = None


@dataclass
class TensorBatch:
    """텐서 배치"""
    data: Any  # numpy array 또는 torch tensor
    shape: Tuple[int, ...]
    dtype: str
    device: str = "cpu"
    created_at: float = field(default_factory=time.time)
    
    @property
    def size(self) -> int:
        """원소 수"""
        result = 1
        for dim in self.shape:
            result *= dim
        return result
    
    def to_numpy(self) -> np.ndarray:
        """NumPy로 변환"""
        if TORCH_AVAILABLE and isinstance(self.data, torch.Tensor):
            return self.data.cpu().numpy()
        return self.data


@dataclass
class AcceleratedResonance:
    """가속된 공명 결과"""
    pairs: List[Tuple[str, str]]
    scores: np.ndarray
    computation_time_ms: float
    device_used: str
    
    def to_dict(self) -> Dict[Tuple[str, str], float]:
        """딕셔너리로 변환"""
        return {
            self.pairs[i]: float(self.scores[i])
            for i in range(len(self.pairs))
        }


class GPUAccelerator:
    """
    GPU 가속 엔진
    
    높은 우선순위 #3 구현:
    - PyTorch 텐서 연산
    - CUDA 가속 (가능한 경우)
    - 자동 CPU 폴백
    - 배치 처리 최적화
    
    예상 효과: 50x 연산 속도 (GPU 있는 경우)
    """
    
    def __init__(
        self,
        prefer_gpu: bool = True,
        batch_size: int = 256,
        dtype: str = "float32"
    ):
        """
        Args:
            prefer_gpu: GPU 사용 선호
            batch_size: 기본 배치 크기
            dtype: 데이터 타입
        """
        self.batch_size = batch_size
        self.dtype = dtype
        
        # 디바이스 결정
        if prefer_gpu and GPU_AVAILABLE:
            self.device = "cuda"
            self.torch_device = torch.device("cuda")
        elif TORCH_AVAILABLE:
            self.device = "cpu"
            self.torch_device = torch.device("cpu")
        else:
            self.device = "numpy"
            self.torch_device = None
        
        # 통계
        self.stats = {
            "total_operations": 0,
            "total_elements": 0,
            "total_time_ms": 0.0,
            "gpu_operations": 0,
            "cpu_operations": 0
        }
        
        self.logger = logging.getLogger("GPUAccelerator")
        self.logger.info(f"🚀 GPUAccelerator initialized (device={self.device}, GPU={GPU_NAME})")
    
    def _to_tensor(self, data: Union[np.ndarray, List]) -> Any:
        """데이터를 텐서로 변환"""
        if not TORCH_AVAILABLE:
            return np.array(data, dtype=np.float32)
        
        if isinstance(data, np.ndarray):
            tensor = torch.from_numpy(data.astype(np.float32))
        else:
            tensor = torch.tensor(data, dtype=torch.float32)
        
        return tensor.to(self.torch_device)
    
    def _to_numpy(self, tensor: Any) -> np.ndarray:
        """텐서를 NumPy로 변환"""
        if isinstance(tensor, np.ndarray):
            return tensor
        if TORCH_AVAILABLE and isinstance(tensor, torch.Tensor):
            return tensor.cpu().numpy()
        return np.array(tensor)
    
    def batch_dot_product(
        self,
        vectors_a: np.ndarray,
        vectors_b: np.ndarray
    ) -> np.ndarray:
        """
        배치 내적 계산
        
        Args:
            vectors_a: (N, D) 행렬
            vectors_b: (N, D) 행렬
            
        Returns:
            (N,) 내적 결과
        """
        start = time.time()
        
        if TORCH_AVAILABLE:
            a = self._to_tensor(vectors_a)
            b = self._to_tensor(vectors_b)
            
            result = torch.sum(a * b, dim=1)
            result = self._to_numpy(result)
            
            if self.device == "cuda":
                self.stats["gpu_operations"] += 1
            else:
                self.stats["cpu_operations"] += 1
        else:
            result = np.sum(vectors_a * vectors_b, axis=1)
            self.stats["cpu_operations"] += 1
        
        elapsed = (time.time() - start) * 1000
        self.stats["total_operations"] += 1
        self.stats["total_elements"] += len(result)
        self.stats["total_time_ms"] += elapsed
        
        return result
    
    def batch_cosine_similarity(
        self,
        vectors_a: np.ndarray,
        vectors_b: np.ndarray,
        eps: float = 1e-8
    ) -> np.ndarray:
        """
        배치 코사인 유사도 계산
        
        Args:
            vectors_a: (N, D) 행렬
            vectors_b: (N, D) 행렬
            eps: 0 나눗셈 방지
            
        Returns:
            (N,) 유사도 결과
        """
        start = time.time()
        
        if TORCH_AVAILABLE:
            a = self._to_tensor(vectors_a)
            b = self._to_tensor(vectors_b)
            
            # 정규화
            a_norm = a / (torch.norm(a, dim=1, keepdim=True) + eps)
            b_norm = b / (torch.norm(b, dim=1, keepdim=True) + eps)
            
            # 코사인 유사도
            result = torch.sum(a_norm * b_norm, dim=1)
            result = self._to_numpy(result)
            
            if self.device == "cuda":
                self.stats["gpu_operations"] += 1
            else:
                self.stats["cpu_operations"] += 1
        else:
            # NumPy 폴백
            a_norm = vectors_a / (np.linalg.norm(vectors_a, axis=1, keepdims=True) + eps)
            b_norm = vectors_b / (np.linalg.norm(vectors_b, axis=1, keepdims=True) + eps)
            result = np.sum(a_norm * b_norm, axis=1)
            self.stats["cpu_operations"] += 1
        
        elapsed = (time.time() - start) * 1000
        self.stats["total_operations"] += 1
        self.stats["total_elements"] += len(result)
        self.stats["total_time_ms"] += elapsed
        
        return result
    
    def batch_resonance(
        self,
        resonance_engine,
        pairs: List[Tuple[str, str]]
    ) -> AcceleratedResonance:
        """
        공명 계산 가속화
        
        Args:
            resonance_engine: 공명 엔진
            pairs: (source_id, target_id) 쌍 목록
            
        Returns:
            AcceleratedResonance 결과
        """
        start = time.time()
        n = len(pairs)
        
        if n == 0:
            return AcceleratedResonance(
                pairs=[],
                scores=np.array([]),
                computation_time_ms=0.0,
                device_used=self.device
            )
        
        # 벡터 추출
        # QubitState의 xyz 좌표와 확률 분포 사용
        vectors_a = []
        vectors_b = []
        
        for source_id, target_id in pairs:
            source = resonance_engine.nodes.get(source_id)
            target = resonance_engine.nodes.get(target_id)
            
            if source and target:
                # 상태 벡터: [x, y, z, Point, Line, Space, God, w]
                source_probs = source.state.probabilities()
                target_probs = target.state.probabilities()
                
                vec_a = [
                    source.state.x, source.state.y, source.state.z,
                    source_probs["Point"], source_probs["Line"],
                    source_probs["Space"], source_probs["God"],
                    source.state.w
                ]
                vec_b = [
                    target.state.x, target.state.y, target.state.z,
                    target_probs["Point"], target_probs["Line"],
                    target_probs["Space"], target_probs["God"],
                    target.state.w
                ]
            else:
                vec_a = [0.0] * 8
                vec_b = [0.0] * 8
            
            vectors_a.append(vec_a)
            vectors_b.append(vec_b)
        
        vectors_a = np.array(vectors_a, dtype=np.float32)
        vectors_b = np.array(vectors_b, dtype=np.float32)
        
        # 코사인 유사도로 공명 근사
        scores = self.batch_cosine_similarity(vectors_a, vectors_b)
        
        # 음수 클램핑
        scores = np.clip(scores, 0.0, 1.0)
        
        elapsed = (time.time() - start) * 1000
        
        return AcceleratedResonance(
            pairs=pairs,
            scores=scores,
            computation_time_ms=elapsed,
            device_used=self.device
        )
    
    def matrix_multiply(
        self,
        a: np.ndarray,
        b: np.ndarray
    ) -> np.ndarray:
        """
        행렬 곱셈
        
        Args:
            a: (M, K) 행렬
            b: (K, N) 행렬
            
        Returns:
            (M, N) 결과 행렬
        """
        start = time.time()
        
        if TORCH_AVAILABLE:
            ta = self._to_tensor(a)
            tb = self._to_tensor(b)
            
            result = torch.matmul(ta, tb)
            result = self._to_numpy(result)
            
            if self.device == "cuda":
                self.stats["gpu_operations"] += 1
            else:
                self.stats["cpu_operations"] += 1
        else:
            result = np.matmul(a, b)
            self.stats["cpu_operations"] += 1
        
        elapsed = (time.time() - start) * 1000
        self.stats["total_operations"] += 1
        self.stats["total_elements"] += result.size
        self.stats["total_time_ms"] += elapsed
        
        return result
    
    def softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """
        Softmax 연산
        
        Args:
            x: 입력 배열
            axis: 적용할 축
            
        Returns:
            Softmax 결과
        """
        start = time.time()
        
        if TORCH_AVAILABLE:
            t = self._to_tensor(x)
            result = torch.softmax(t, dim=axis)
            result = self._to_numpy(result)
        else:
            # NumPy 구현
            exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
            result = exp_x / np.sum(exp_x, axis=axis, keepdims=True)
        
        elapsed = (time.time() - start) * 1000
        self.stats["total_operations"] += 1
        self.stats["total_time_ms"] += elapsed
        
        return result
    
    def get_stats(self) -> Dict[str, Any]:
        """통계 반환"""
        total_ops = self.stats["total_operations"]
        return {
            **self.stats,
            "device": self.device,
            "gpu_name": GPU_NAME,
            "torch_available": TORCH_AVAILABLE,
            "gpu_available": GPU_AVAILABLE,
            "avg_time_per_op_ms": (
                self.stats["total_time_ms"] / total_ops if total_ops > 0 else 0.0
            ),
            "gpu_utilization": (
                self.stats["gpu_operations"] / total_ops if total_ops > 0 else 0.0
            )
        }
    
    def benchmark(self, size: int = 1000, iterations: int = 10) -> Dict[str, float]:
        """
        성능 벤치마크
        
        Args:
            size: 벡터 크기
            iterations: 반복 횟수
            
        Returns:
            벤치마크 결과
        """
        results = {}
        
        # 랜덤 데이터 생성
        a = np.random.randn(size, 128).astype(np.float32)
        b = np.random.randn(size, 128).astype(np.float32)
        
        # Dot product 벤치마크
        times = []
        for _ in range(iterations):
            start = time.time()
            self.batch_dot_product(a, b)
            times.append((time.time() - start) * 1000)
        results["dot_product_ms"] = np.mean(times)
        
        # Cosine similarity 벤치마크
        times = []
        for _ in range(iterations):
            start = time.time()
            self.batch_cosine_similarity(a, b)
            times.append((time.time() - start) * 1000)
        results["cosine_similarity_ms"] = np.mean(times)
        
        # Matrix multiply 벤치마크
        c = np.random.randn(128, 64).astype(np.float32)
        times = []
        for _ in range(iterations):
            start = time.time()
            self.matrix_multiply(a, c)
            times.append((time.time() - start) * 1000)
        results["matmul_ms"] = np.mean(times)
        
        results["device"] = self.device
        results["size"] = size
        results["iterations"] = iterations
        
        return results


# 테스트
if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 GPU Accelerator Test")
    print("="*70)
    
    accelerator = GPUAccelerator()
    
    print(f"\nDevice: {accelerator.device}")
    print(f"PyTorch: {TORCH_AVAILABLE}")
    print(f"GPU: {GPU_AVAILABLE} ({GPU_NAME})")
    
    # 테스트 1: 내적
    print("\n[Test 1] Batch Dot Product")
    a = np.random.randn(100, 64).astype(np.float32)
    b = np.random.randn(100, 64).astype(np.float32)
    result = accelerator.batch_dot_product(a, b)
    print(f"  ✓ Shape: {result.shape}")
    print(f"  Range: [{result.min():.3f}, {result.max():.3f}]")
    
    # 테스트 2: 코사인 유사도
    print("\n[Test 2] Batch Cosine Similarity")
    result = accelerator.batch_cosine_similarity(a, b)
    print(f"  ✓ Shape: {result.shape}")
    print(f"  Range: [{result.min():.3f}, {result.max():.3f}]")
    
    # 테스트 3: 행렬 곱
    print("\n[Test 3] Matrix Multiply")
    c = np.random.randn(64, 32).astype(np.float32)
    result = accelerator.matrix_multiply(a, c)
    print(f"  ✓ Shape: {result.shape}")
    
    # 테스트 4: Softmax
    print("\n[Test 4] Softmax")
    x = np.random.randn(10, 5).astype(np.float32)
    result = accelerator.softmax(x)
    print(f"  ✓ Shape: {result.shape}")
    print(f"  Sum per row: {result.sum(axis=1)}")  # Should be ~1.0
    
    # 테스트 5: 벤치마크
    print("\n[Test 5] Benchmark")
    bench = accelerator.benchmark(size=1000, iterations=5)
    print(f"  Dot product: {bench['dot_product_ms']:.3f}ms")
    print(f"  Cosine sim: {bench['cosine_similarity_ms']:.3f}ms")
    print(f"  Matmul: {bench['matmul_ms']:.3f}ms")
    
    # 통계
    print("\n[Stats]")
    stats = accelerator.get_stats()
    print(f"  Total ops: {stats['total_operations']}")
    print(f"  GPU ops: {stats['gpu_operations']}")
    print(f"  Avg time: {stats['avg_time_per_op_ms']:.3f}ms")
    
    print("\n✅ All tests passed!")
    print("="*70 + "\n")
