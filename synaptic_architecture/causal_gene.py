import numpy as np
from typing import Dict, List, Any

class GeneticSynthesizer:
    """
    [Synaptic Architecture] Autonomous Genetic Evolution
    정보의 파편(Bit-Genes)들을 교차(Crossover)하고 변이(Mutation)시켜
    새로운 '논리적 종(Logical Species)'을 스스로 번식시킵니다.
    클래스라는 고정된 틀을 파괴하고, 유동적인 유전자 풀(Gene Pool)을 형성합니다.
    """
    def __init__(self):
        self.gene_pool = {} # Name -> Bit-Gene (uint64)

    def synthesize(self, parent_a: np.uint64, parent_b: np.uint64) -> np.uint64:
        """
        [Crossover] 두 유전 정보를 섞어 새로운 논리를 탄생시킵니다.
        """
        mask = np.uint64(0xFFFFFFFF00000000)
        # 상위 32비트와 하위 32비트를 교차 결합
        child = (parent_a & mask) | (parent_b & ~mask)

        # [Mutation] 낮은 확률로 비트 변이 발생 (영감/노이즈)
        if np.random.random() < 0.05:
            mutation_bit = np.uint64(1 << np.random.randint(0, 64))
            child ^= mutation_bit

        return child

    def evolve_principles(self, field_state: Dict[str, Any]):
        """
        장내의 보텍스(Vortices)들을 부모로 삼아 새로운 유전자를 합성합니다.
        """
        vortices = field_state.get("detected_vortices", [])
        if len(vortices) < 2: return

        v1_gene = np.uint64(int(vortices[0]['resonant_gene'], 16))
        v2_gene = np.uint64(int(vortices[1]['resonant_gene'], 16))

        new_gene = self.synthesize(v1_gene, v2_gene)
        gene_name = f"GENE_{hex(new_gene)}"

        self.gene_pool[gene_name] = new_gene
        print(f"[Genetic Synthesis] New Logical Species evolved: {gene_name}")

    def get_active_genes(self) -> List[np.uint64]:
        return list(self.gene_pool.values())
