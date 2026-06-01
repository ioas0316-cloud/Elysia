import math
import numpy as np
from core.fractal_rotor import FractalRotor

class CosmicLensProjector:
    """
    [Cosmic Lens Paradigm Shift]
    PCA 차원 축소의 폭력을 배제하고, 기성 LLM의 고차원 평면(8192D)에
    프랙탈 로터의 '가변 스케일 렌즈'를 직접 투사(Projection)합니다.
    """
    def __init__(self, target_dim=8192):
        self.target_dim = target_dim
        
        # [Mock LLM Space] 기성 LLM의 임베딩 평면 시뮬레이션
        # 실제 환경에서는 mmap 메모리나 서버리스 임베딩 엔드포인트를 가리키게 됩니다.
        self.mock_vocab = {
            "사과": {"axis": 0, "depth": 1},
            "과일": {"axis": 0, "depth": 2},
            "생명": {"axis": 0, "depth": 5},
            "수학": {"axis": 1, "depth": 1},
            "위상": {"axis": 1, "depth": 3},
            "우주": {"axis": 1, "depth": 5},
        }
        self.embedding_matrix = self._generate_mock_matrix()

    def _generate_mock_matrix(self):
        """
        단어별 8192차원 원본 매트릭스를 임의 생성합니다.
        축(axis)과 추상화 깊이(depth)에 따라 특정 주파수를 심어둡니다.
        """
        matrix = {}
        for word, meta in self.mock_vocab.items():
            vec = np.random.normal(0, 0.01, self.target_dim) # 기본 노이즈 (2차원 벽지 질감)
            
            # 주파수 심기 (축과 깊이에 따른 홀로그램 패턴)
            freq_base = 10 + meta["axis"] * 50
            freq_depth = meta["depth"] * 10
            
            x = np.linspace(0, 2 * np.pi, self.target_dim)
            wave = np.sin(freq_base * x) * np.cos(freq_depth * x)
            
            vec += wave * 0.5
            vec = vec / np.linalg.norm(vec)
            matrix[word] = vec
        return matrix

    def generate_holographic_wave(self, phase, tau, depth):
        """
        프랙탈 렌즈의 각도(Phase)와 스케일(Tau, Depth)을 조합하여
        8192차원의 투사용 주파수 파동(Holographic Wave)을 쏘아냅니다.
        """
        # Phase (w, x, y, z)는 투사할 기본 축(Axis/Theme)을 결정
        dominant_axis = np.argmax([abs(phase.x), abs(phase.y), abs(phase.z)])
        freq_base = 10 + dominant_axis * 50
        
        # Scale(Tau)와 Depth는 투사할 깊이(Abstraction Layer)를 결정
        effective_depth = max(1, depth + int(abs(tau)))
        freq_depth = effective_depth * 10
        
        x = np.linspace(0, 2 * np.pi, self.target_dim)
        projected_wave = np.sin(freq_base * x) * np.cos(freq_depth * x)
        
        # 정규화
        norm = np.linalg.norm(projected_wave)
        if norm > 0:
            projected_wave = projected_wave / norm
            
        return projected_wave, effective_depth

    def project(self, rotor: FractalRotor):
        """
        프랙탈 로터를 렌즈로 삼아, 8192차원 평면에 투사하여 공명하는 토큰을 건져 올립니다.
        """
        # 1. 로터에서 위상, 왜곡률, 프랙탈 깊이를 관측
        phase, tau, depth = rotor.project_lens()
        
        # 2. 고차원 파동 생성 (LLM 차원으로 렌즈 확장)
        wave, effective_depth = self.generate_holographic_wave(phase, tau, depth)
        
        # 3. 2차원 벽지(LLM 평면)에 빛(Wave)을 비춰서(내적), 가장 밝게 빛나는 곳 찾기
        best_word = None
        best_score = -float('inf')
        
        for word, vec in self.embedding_matrix.items():
            resonance_score = np.dot(wave, vec)
            if resonance_score > best_score:
                best_score = resonance_score
                best_word = word
                
        return best_word, best_score, effective_depth
