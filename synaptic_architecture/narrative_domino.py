import numpy as np
from typing import List, Dict, Any, Tuple
from .bit_logic import BitInterference
from .field import CrystallizationField

class NarrativeDominoKernel:
    """
    [Narrative Domino Kernel]
    "이미 같은 것들이라면 계산과 판단은 불필요해진다."

    이 커널은 텐서 연산을 통하지 않고, 비트-유전자의 공명(Resonance)만으로
    정보를 즉각적으로 정렬하고 연결하는 '서사적 도미노'를 집행합니다.
    """
    def __init__(self, field: CrystallizationField):
        self.field = field
        self.interference = BitInterference()

    def process_narrative(self, gene_a: np.uint64, gene_b: np.uint64) -> float:
        """
        두 유전자의 서사적 공명도를 측정합니다.
        XOR 상쇄가 일어나는 지점이 '도미노'의 시작점입니다.
        """
        return self.interference.interference_score(gene_a, gene_b)

    def trigger_domino(self, input_gene: np.uint64, start_pos: np.ndarray):
        """
        입력된 유전자가 필드에 닿는 순간, 공명하는 주변 유전자들을
        연쇄적으로 활성화(Domino Effect)시킵니다.
        """
        # 1. 초기 활성화 주입
        self.field.inject_activation(start_pos, 10.0)

        # 2. 필드 전체에서 공명 지점 탐색 (O(1) 지향형 비트 비교)
        # 실제로는 CrystallizationField의 bit_genes를 배치(Batch)로 비교
        diff = np.bitwise_xor(self.field.bit_genes, input_gene)

        # 비트 카운팅 (64비트)
        def bit_count(n):
            n = (n & 0x5555555555555555) + ((n >> 1) & 0x5555555555555555)
            n = (n & 0x3333333333333333) + ((n >> 2) & 0x3333333333333333)
            n = (n & 0x0F0F0F0F0F0F0F0F) + ((n >> 4) & 0x0F0F0F0F0F0F0F0F)
            n = (n & 0x00FF00FF00FF00FF) + ((n >> 8) & 0x00FF00FF00FF00FF)
            n = (n & 0x0000FFFF0000FFFF) + ((n >> 16) & 0x0000FFFF0000FFFF)
            n = (n & 0x00000000FFFFFFFF) + ((n >> 32) & 0x00000000FFFFFFFF)
            return n

        deficit = bit_count(diff)
        resonance_map = 1.0 - (deficit / 64.0)

        # 3. 공명 임계치를 넘는 지점들을 '도미노'로 활성화
        # 계산이 아니라 '자석처럼 끌리는' 물리적 현상을 시뮬레이션
        domino_mask = resonance_map > 0.8
        self.field.activation[domino_mask] += resonance_map[domino_mask] * 5.0

        # 4. 필드 전파 (Propagation을 통한 서사적 흐름 완성)
        self.field.propagate(decay=0.98, spreading_factor=0.7)

    def align_narratives(self, genes: List[np.uint64]) -> np.ndarray:
        """
        여러 유전자들이 주어졌을 때, 이들이 필드 위에서
        어떻게 정렬되는지(Resonance Center)를 반환합니다.
        """
        # 가장 강력한 공명 중심(Vortex)을 찾습니다.
        if not genes: return np.array([0, 0])

        # 대표 유전자 선정 (첫 번째 유전자를 시드로 사용)
        seed_gene = genes[0]

        # 필드 정렬 시뮬레이션 (단순화)
        # 실제로는 모든 유전자가 서로를 끌어당기며 평형점에 도달
        res_map = np.zeros((self.field.resolution, self.field.resolution))
        for g in genes:
            diff = np.bitwise_xor(self.field.bit_genes, g)
            # ... 비트 카운트 및 resonance 계산 (생략형)
            # 여기서는 단순 합산으로 표현

        # 최고 활성화 지점이 서사의 '중심핵'
        idx = np.argmax(self.field.activation)
        return np.unravel_index(idx, self.field.activation.shape)

if __name__ == "__main__":
    cf = CrystallizationField(resolution=64)
    kernel = NarrativeDominoKernel(cf)

    apple_gene = np.uint64(0x1234567812345678)
    banana_gene = np.uint64(0x1234567800000000) # 과일 계보(상위 32비트) 공유

    print(f"Apple-Banana Resonance: {kernel.process_narrative(apple_gene, banana_gene):.4f}")

    kernel.trigger_domino(apple_gene, np.array([32, 32]))
    print("Domino triggered.")
