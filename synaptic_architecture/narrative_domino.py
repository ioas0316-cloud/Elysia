import numpy as np
from typing import List, Dict, Any, Tuple
from core.utils.math_utils import popcount_vectorized
from .bit_logic import BitInterference
from .field import CrystallizationField

class NarrativeDominoKernel:
    """
    [Narrative Domino Kernel v2.0]
    "스케일에 상관없이 정보는 서사의 흐름을 타고 즉각 전도된다."

    미시적 비트의 상쇄뿐만 아니라, 거시적 패턴(Macro)의 일치가
    하위의 수많은 노드들을 한꺼번에 쓰러뜨리는 '거대한 도미노'를 집행합니다.
    """
    def __init__(self, field: CrystallizationField):
        self.field = field
        self.interference = BitInterference()

    def trigger_hierarchical_domino(self, genes: Dict[str, np.uint64], start_pos: np.ndarray):
        """
        상위(Macro) 유전자가 일치하면, 필드의 광범위한 영역에 걸쳐
        서사적 활성화(Domino)를 즉각 폭발시킵니다.
        """
        # 1. 거시적 공명 맵 생성 (Macro Resonance)
        macro_gene = genes.get("macro", np.uint64(0))

        # 필드 내의 모든 유전자들과 macro 수준에서 비교
        macro_diff = np.bitwise_xor(self.field.bit_genes, macro_gene)
        macro_res = 1.0 - (popcount_vectorized(macro_diff) / 64.0)

        # 2. 거대한 도미노 집행
        macro_threshold = 0.7
        self.field.activation += (macro_res > macro_threshold) * 20.0

        # 3. 미시적 정밀 정렬 (Micro Resonance)
        micro_gene = genes.get("micro", np.uint64(0))
        micro_diff = np.bitwise_xor(self.field.bit_genes, micro_gene)
        micro_res = 1.0 - (popcount_vectorized(micro_diff) / 64.0)

        self.field.activation += (micro_res > 0.9) * 10.0

        # 4. 서사적 장력 전파
        self.field.propagate(decay=0.95, spreading_factor=0.8)

    def find_resonance_vortex(self, genes: Dict[str, np.uint64]) -> np.ndarray:
        """가장 강력하게 모든 계층이 공명하는 소용돌이 지점을 찾습니다."""
        idx = np.argmax(self.field.activation)
        return np.unravel_index(idx, self.field.activation.shape)

    def process_narrative(self, gene_a: np.uint64, gene_b: np.uint64) -> float:
        """Legacy support for simple bit-gene comparison."""
        diff = gene_a ^ gene_b
        from core.utils.math_utils import popcount
        return 1.0 - (popcount(diff) / 64.0)
