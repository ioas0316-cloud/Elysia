try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

import math

if HAS_TORCH:
    class HangulKroneckerTokenizer(nn.Module):
        """
        한글 자모 분해 및 2x2 Kronecker Matrix Product 기반
        64차원 Topological Phase Wavevector Generator (PyTorch Version)
        """
        def __init__(self, target_dim=64):
            super().__init__()
            assert target_dim == 64, "Kronecker (2x2)^3 expansion yields exactly 64 dimensions."
            self.target_dim = target_dim

            # 초성(19), 중성(21), 종성(28) 위상 각도 매핑 임베딩
            self.cho_angles = nn.Parameter(torch.linspace(0, 2 * math.pi, 19))
            self.jung_angles = nn.Parameter(torch.linspace(0, 2 * math.pi, 21))
            self.jong_angles = nn.Parameter(torch.linspace(0, 2 * math.pi, 28))

        def _char_to_jamo_indices(self, char: str):
            """한글 음절을 (초성, 중성, 종성) 인덱스로 분해"""
            code = ord(char) - 0xAC00
            if code < 0 or code > 11171:
                return None # 한글 이외 문자는 예외 처리

            cho = code // 588
            jung = (code % 588) // 28
            jong = code % 28
            return cho, jung, jong

        def _build_2x2_rotation_matrix(self, theta: torch.Tensor) -> torch.Tensor:
            """2x2 Fundamental Phase Matrix 생성"""
            cos_t = torch.cos(theta)
            sin_t = torch.sin(theta)
            return torch.stack([
                torch.stack([cos_t, -sin_t]),
                torch.stack([sin_t,  cos_t])
            ]) # [2, 2]

        def kronecker_product_2x2(self, A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
            """두 [2, 2] 행렬의 Kronecker Product -> [4, 4]"""
            return torch.einsum('ij,kl->ikjl', A, B).reshape(4, 4)

        def kronecker_product_4x2(self, A_4x4: torch.Tensor, B_2x2: torch.Tensor) -> torch.Tensor:
            """[4, 4]와 [2, 2] 행렬의 Kronecker Product -> [8, 8]"""
            return torch.einsum('ij,kl->ikjl', A_4x4, B_2x2).reshape(8, 8)

        def forward(self, text: str) -> torch.Tensor:
            """
            입력 텍스트를 64차원 SDF Wavevector Stream으로 변환 [Sequence_Len, 64]
            """
            wavevectors = []
            for char in text:
                indices = self._char_to_jamo_indices(char)
                if indices is None:
                    # 공백/특수문자: Zero-Phase Wavevector
                    wavevectors.append(torch.zeros(self.target_dim))
                    continue

                cho_idx, jung_idx, jong_idx = indices

                # 1. 자모별 2x2 회전 행렬 도출
                M_cho = self._build_2x2_rotation_matrix(self.cho_angles[cho_idx])
                M_jung = self._build_2x2_rotation_matrix(self.jung_angles[jung_idx])
                M_jong = self._build_2x2_rotation_matrix(self.jong_angles[jong_idx])

                # 2. 3중 Kronecker Product 수행: (2x2) ⊗ (2x2) ⊗ (2x2) = (8x8) = 64 elements
                M_4x4 = self.kronecker_product_2x2(M_cho, M_jung)
                M_8x8 = self.kronecker_product_4x2(M_4x4, M_jong)

                # 3. 64차원 위상 파수 벡터로 평탄화 (Flatten)
                k_text_vec = M_8x8.reshape(-1)
                wavevectors.append(k_text_vec)

            return torch.stack(wavevectors) # [Seq_len, 64]

else:
    import numpy as np

    class HangulKroneckerTokenizer:
        """
        한글 자모 분해 및 2x2 Kronecker Matrix Product 기반
        64차원 Topological Phase Wavevector Generator (NumPy Fallback Version)
        """
        def __init__(self, target_dim=64):
            assert target_dim == 64, "Kronecker (2x2)^3 expansion yields exactly 64 dimensions."
            self.target_dim = target_dim

            self.cho_angles = np.linspace(0, 2 * math.pi, 19)
            self.jung_angles = np.linspace(0, 2 * math.pi, 21)
            self.jong_angles = np.linspace(0, 2 * math.pi, 28)

        def _char_to_jamo_indices(self, char: str):
            code = ord(char) - 0xAC00
            if code < 0 or code > 11171:
                return None

            cho = code // 588
            jung = (code % 588) // 28
            jong = code % 28
            return cho, jung, jong

        def _build_2x2_rotation_matrix(self, theta: float) -> np.ndarray:
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            return np.array([
                [cos_t, -sin_t],
                [sin_t,  cos_t]
            ], dtype=np.float32)

        def forward(self, text: str) -> np.ndarray:
            wavevectors = []
            for char in text:
                indices = self._char_to_jamo_indices(char)
                if indices is None:
                    wavevectors.append(np.zeros(self.target_dim, dtype=np.float32))
                    continue

                cho_idx, jung_idx, jong_idx = indices

                M_cho = self._build_2x2_rotation_matrix(self.cho_angles[cho_idx])
                M_jung = self._build_2x2_rotation_matrix(self.jung_angles[jung_idx])
                M_jong = self._build_2x2_rotation_matrix(self.jong_angles[jong_idx])

                M_4x4 = np.kron(M_cho, M_jung)
                M_8x8 = np.kron(M_4x4, M_jong)

                wavevectors.append(M_8x8.reshape(-1))

            return np.array(wavevectors, dtype=np.float32)

        def __call__(self, text: str):
            return self.forward(text)

# ==========================================
# 한글 토크나이저 실행 시뮬레이션
# ==========================================
if __name__ == "__main__":
    tokenizer = HangulKroneckerTokenizer()
    sample_text = "엘리시아"
    k_text_stream = tokenizer(sample_text)

    print("=== Topological Hangul Jamo Kronecker Tokenizer ===")
    print(f"Input Text: '{sample_text}'")
    print(f"Generated Wavevector Stream Shape: {k_text_stream.shape}") # [4, 64]
    if HAS_TORCH:
        norm_val = torch.norm(k_text_stream[0]).item()
    else:
        norm_val = np.linalg.norm(k_text_stream[0])
    print(f"Sample Wavevector Norm (Eikonal Preserved): {norm_val:.4f}")
