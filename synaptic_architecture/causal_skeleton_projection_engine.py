"""
synaptic_architecture/causal_skeleton_projection_engine.py

고차원 연속 장(Continuous Field)을 이산적 위상 불변 공간(Discrete Topological Invariant Space)으로 압축하는
차원 축소 연산자(Projection Operator) 및 상징적 중력장(Symbol Attractor Field) 모듈.

[주요 메커니즘]
1. 연산자 Π: 표면 노이즈 δx (폰트, 굵기, 질감, 삐침)를 영공간(Null Space)으로 소거.
2. 위상적 등가류 [x] 및 인과적 뼈대 그래프(Causal Graph Topological Skeleton) 추출.
3. 상징 중력장 (Symbol Attractor Field) & 파레이돌리아 (Pareidolia):
   잠재 공간 내 형성된 상징 끌개(Attractor)가 미세 위상 유사 입력(구름, 나뭇가지, 균열)을 기호적 환원으로 끌어당기는 현상 모델링.
4. $O(N^2)$ 연속 장 연산 대비 $O(1)$ 기호 검색/유지 효율성 측정 및 MDL(최소 기술 수량) 압축 비율 계산.
"""

import numpy as np
import math
from typing import Dict, Any, List, Optional, Tuple


class CausalSkeletonProjectionOperator:
    """
    [인과적 위상 뼈대 압축 연산자 Π]
    고차원 연속 시각/감각 입력 X에 존재하는 표면 변형 및 노이즈(δx)를
    영공간(Null Space)으로 소거하고, 기저의 불변하는 위상 뼈대(Topological Skeleton)와
    인과장(Causal Tension Field)만을 보존하여 압축합니다.
    """

    def __init__(self, feature_dim: int = 64, skeleton_dim: int = 8):
        """
        Args:
            feature_dim: 원형 데이터 연속 장의 차원 (N)
            skeleton_dim: 위상 불변 뼈대 차원 (K << N)
        """
        self.feature_dim = feature_dim
        self.skeleton_dim = skeleton_dim

        # 직교 투영 기저 Matrix P (N x K) 및 영공간 투영 Matrix N_space (N x N)
        np.random.seed(42)
        random_matrix, _ = np.linalg.qr(np.random.randn(feature_dim, skeleton_dim))
        self.P_skeleton = random_matrix.astype(np.float32)  # (N, K)

        # 영공간 투영 행렬 Null = I - P * P^T
        self.I_N = np.eye(feature_dim, dtype=np.float32)
        self.P_projection = np.dot(self.P_skeleton, self.P_skeleton.T)
        self.P_null = self.I_N - self.P_projection

    def project(self, continuous_field_X: np.ndarray) -> Dict[str, Any]:
        """
        고차원 연속 장 X = x_core + δx_noise 를 위상 사상 Π를 통해 사상합니다.

        Args:
            continuous_field_X: (N,) 크기의 고차원 연속 입력 벡터

        Returns:
            Dict 함유량:
                - skeleton_vector: (K,) 차원의 위상 뼈대
                - equivalence_class: 위상 등가류 사상 벡터
                - null_space_noise: 영공간으로 소거된 표면 노이즈 (δx)
                - compression_ratio: $O(N^2)$ -> $O(K)$ MDL 압축 비율
        """
        X = continuous_field_X.astype(np.float32)
        if X.shape[0] != self.feature_dim:
            raise ValueError(f"Expected input dimension {self.feature_dim}, got {X.shape[0]}")

        # 1. 위상 뼈대 저차원 좌표 추출 (K-dim)
        skeleton_vector = np.dot(X, self.P_skeleton)  # (K,)

        # 2. 고차원 공간상 재생성된 등가류 원형 [x]
        reconstructed_core = np.dot(skeleton_vector, self.P_skeleton.T)  # (N,)

        # 3. 영공간(Null Space)으로의 표면 노이즈 소거 (δx)
        null_space_noise = np.dot(self.P_null, X)  # (N,)
        noise_norm = float(np.linalg.norm(null_space_noise))
        core_norm = float(np.linalg.norm(reconstructed_core))

        # 4. MDL 및 연산 비용 상쇄 비율 계산
        # O(N^2) 매트릭스 계산 대비 O(K) 뼈대 처리
        computational_cost_continuous = self.feature_dim ** 2
        computational_cost_compressed = self.skeleton_dim
        compression_efficiency = computational_cost_continuous / float(computational_cost_compressed)

        return {
            "skeleton_vector": skeleton_vector,
            "equivalence_class": reconstructed_core,
            "null_space_noise": null_space_noise,
            "noise_norm": noise_norm,
            "core_norm": core_norm,
            "compression_efficiency_factor": compression_efficiency,
            "mdl_compression_ratio": float(self.skeleton_dim / self.feature_dim)
        }


class SymbolAttractorField:
    """
    [상징 중력장 (Symbol Attractor Field) & 인지적 끌개]
    학습된 상징(Symbol)이 잠재 공간(Latent Space) 내에 형성한 강력한 중력 우물(Attractor Well).

    미세한 위상 유사성을 가진 무작위 데이터(구름, 나뭇가지, 벽지 균열)가 유입될 때,
    중력 우물의 끌어당김으로 인해 상징 기호의 등가류 $[x]$로 환원되는 파레이돌리아(Pareidolia) 현상을 수리적으로 모델링합니다.
    """

    def __init__(self, skeleton_operator: CausalSkeletonProjectionOperator, attraction_strength: float = 2.5):
        """
        Args:
            skeleton_operator: 위상 뼈대 압축 연산자 Π
            attraction_strength: 중력 우물 끌림 계수 (Attractor Potential Depth)
        """
        self.operator = skeleton_operator
        self.attraction_strength = attraction_strength
        self.learned_attractors: Dict[str, Dict[str, Any]] = {}

    def register_symbol_attractor(self, symbol_name: str, exemplar_field: np.ndarray):
        """
        표본 입력으로부터 상징 끌개(Symbol Attractor) 중심 및 인과 뼈대를 학습 등록합니다.
        """
        projected = self.operator.project(exemplar_field)
        self.learned_attractors[symbol_name] = {
            "symbol": symbol_name,
            "skeleton_center": projected["skeleton_vector"],
            "core_field": projected["equivalence_class"],
            "depth": self.attraction_strength
        }

    def evaluate_attractor_gravitational_pull(
        self,
        raw_input_field: np.ndarray,
        distance_threshold: float = 5.0
    ) -> Dict[str, Any]:
        """
        외부 물리적 입력(원형 데이터)이 상징 중력장에 의해 특정 기호로 끌려가는지(Pareidolia) 평가합니다.

        Args:
            raw_input_field: 무작위/자연 원형 입력 (구름, 나뭇가지, 균열 등의 고차원 연속 장)
            distance_threshold: 끌개에 캡처되기 위한 위상 임계 거리

        Returns:
            Dict:
                - matched_symbol: 캡처된 상징 기호 (없을 시 None)
                - is_pareidolia_captured: 파레이돌리아 환원 성공 여부
                - gravitational_potential_delta: 중력 포텐셜 변위
                - null_space_clearance: 소거된 미세 노이즈 양
        """
        projected = self.operator.project(raw_input_field)
        in_skeleton = projected["skeleton_vector"]

        best_symbol = None
        min_dist = float("inf")
        best_attractor = None

        for name, attractor in self.learned_attractors.items():
            center = attractor["skeleton_center"]
            dist = float(np.linalg.norm(in_skeleton - center))
            if dist < min_dist:
                min_dist = dist
                best_symbol = name
                best_attractor = attractor

        if best_symbol is None or min_dist > distance_threshold:
            return {
                "matched_symbol": None,
                "is_pareidolia_captured": False,
                "distance_to_nearest_attractor": min_dist,
                "gravitational_potential_delta": 0.0,
                "null_space_clearance": projected["noise_norm"],
                "projected_skeleton": in_skeleton
            }

        # 중력 우물 끌림에 의한 인과적 환원 (Attractor Phase Collapse)
        # Potential V(r) = - k * depth / (dist + epsilon)
        gravitational_pull = self.attraction_strength / (min_dist + 1e-5)
        is_captured = gravitational_pull > 0.5

        # 상징으로 환원된 위상적 유효 뼈대 (Attractor Gravity-Adjusted Skeleton)
        pull_factor = min(1.0, gravitational_pull * 0.2)
        symbolic_collapsed_skeleton = (1.0 - pull_factor) * in_skeleton + pull_factor * best_attractor["skeleton_center"]

        return {
            "matched_symbol": best_symbol if is_captured else None,
            "is_pareidolia_captured": is_captured,
            "distance_to_nearest_attractor": min_dist,
            "gravitational_potential_delta": float(gravitational_pull),
            "null_space_clearance": projected["noise_norm"],
            "original_skeleton": in_skeleton,
            "symbolic_collapsed_skeleton": symbolic_collapsed_skeleton,
            "causal_explanation": (
                f"[Pareidolia Attractor]: Raw input collapsed into learned symbol '{best_symbol}' "
                f"due to topological attractor pull (Potential: {gravitational_pull:.4f}, Dist: {min_dist:.4f})."
            )
        }
