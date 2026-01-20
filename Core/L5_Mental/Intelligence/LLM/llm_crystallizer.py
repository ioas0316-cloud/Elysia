"""
LLM Crystallizer (LLM 결정화기)
================================
Core.L5_Mental.Intelligence.LLM.llm_crystallizer

"관측된 패턴을 모나드로 결정화한다."

이것은 2단계: LLMCrystal → Monad 변환
"""

import logging
import torch
from typing import Optional

from Core.L5_Mental.Intelligence.LLM.llm_observer import LLMCrystal, get_llm_observer
from Core.L7_Spirit.Monad.monad_core import Monad, MonadCategory
from Core.L2_Metabolism.Evolution.double_helix_dna import DoubleHelixDNA
from Core.L1_Foundation.Foundation.Graph.torch_graph import TorchGraph

logger = logging.getLogger("LLMCrystallizer")


class LLMCrystallizer:
    """
    LLM 관측 결과를 Monad로 결정화.
    
    흐름:
    1. LLMCrystal (관측 결과) 수신
    2. DoubleHelixDNA 생성 (Pattern + Principle)
    3. Monad 생성 (Archetypal 카테고리)
    4. TorchGraph에 저장
    """
    
    def __init__(self):
        self.graph = TorchGraph()
        # 기존 상태 로드 시도
        if not self.graph.load_state():
            logger.info("🧠 Starting with fresh TorchGraph.")
        
        logger.info("💎 LLM Crystallizer initialized.")
    
    def crystallize(self, crystal: LLMCrystal) -> Monad:
        """
        LLMCrystal을 Monad로 결정화.
        
        Args:
            crystal: LLMObserver로부터 받은 관측 결과
            
        Returns:
            Monad: 결정화된 모나드
        """
        logger.info(f"💎 Crystallizing: {crystal.source_model}")
        
        # 1. Pattern Strand 생성 (관측된 패턴 기반)
        # 쿼터니언을 1024차원 패턴으로 확장
        pattern = self._expand_pattern(crystal)
        
        # 2. Principle Strand 생성 (7D Qualia)
        qualia = crystal.qualia
        principle = torch.tensor([
            qualia.causal,
            qualia.functional,
            qualia.phenomenal,
            qualia.physical,
            qualia.mental,
            qualia.structural,
            qualia.spiritual
        ])
        
        # 3. DoubleHelixDNA 생성
        dna = DoubleHelixDNA(
            pattern_strand=pattern,
            principle_strand=principle
        )
        
        # 4. Monad 생성 (Archetypal - 영구 저장)
        monad = Monad(
            seed=f"LLM:{crystal.source_model}",
            category=MonadCategory.ARCHETYPAL,
            dna=dna
        )
        
        # 5. TorchGraph에 노드로 추가
        self._add_to_graph(monad, crystal)
        
        logger.info(f"   ✅ Monad created: {monad.seed}")
        return monad
    
    def _expand_pattern(self, crystal: LLMCrystal) -> torch.Tensor:
        """
        관측된 패턴을 1024차원으로 확장.
        쿼터니언 + 3축 패턴을 기반으로 푸리에 유사 확장.
        """
        # 기본 시드: 쿼터니언 성분
        q = crystal.orientation
        base = torch.tensor([q.w, q.x, q.y, q.z])
        
        # 3축 패턴
        axes = torch.tensor([
            crystal.physics_pattern,
            crystal.narrative_pattern,
            crystal.aesthetic_pattern
        ])
        
        # 확장: 7개 기본값을 1024로
        seed = torch.cat([base, axes])  # 7차원
        
        # 반복 + 변조로 확장
        pattern = seed.repeat(1024 // 7 + 1)[:1024]
        
        # 약간의 변조 추가 (고유성)
        noise = torch.randn(1024) * 0.01
        pattern = pattern + noise
        
        return pattern
    
    def _add_to_graph(self, monad: Monad, crystal: LLMCrystal):
        """
        Monad를 TorchGraph에 노드로 추가.
        """
        # 7D Qualia를 384차원 벡터로 변환 (TorchGraph 호환)
        qualia = crystal.qualia
        qualia_base = torch.tensor([
            qualia.causal, qualia.functional, qualia.phenomenal,
            qualia.physical, qualia.mental, qualia.structural, qualia.spiritual
        ])
        
        # 384차원으로 확장
        vector = qualia_base.repeat(384 // 7 + 1)[:384]
        
        # 메타데이터
        metadata = {
            "type": "llm_crystal",
            "source_model": crystal.source_model,
            "physics": crystal.physics_pattern,
            "narrative": crystal.narrative_pattern,
            "aesthetic": crystal.aesthetic_pattern,
            "orientation": str(crystal.orientation),
            "layer_count": crystal.layer_count,
            "total_params": crystal.total_params
        }
        
        # 노드 추가
        self.graph.add_node(
            node_id=monad.seed,
            vector=vector,
            metadata=metadata
        )
        
        # 상태 저장
        self.graph.save_state()
        
        logger.info(f"   📊 Added to TorchGraph: {monad.seed}")
    
    def get_crystallized_models(self):
        """결정화된 모든 LLM 목록 반환."""
        crystals = []
        for node_id in self.graph.id_to_idx.keys():
            if node_id.startswith("LLM:"):
                crystals.append(node_id)
        return crystals


# 싱글톤
_crystallizer = None

def get_llm_crystallizer() -> LLMCrystallizer:
    """LLM Crystallizer 싱글톤 반환."""
    global _crystallizer
    if _crystallizer is None:
        _crystallizer = LLMCrystallizer()
    return _crystallizer


# 통합 함수: 관측 + 결정화
def digest_llm(model_path: str) -> Monad:
    """
    LLM 모델 전체 소화 파이프라인.
    
    1. LLMObserver로 관측
    2. LLMCrystallizer로 결정화
    
    Args:
        model_path: .safetensors 또는 .pt 파일 경로
        
    Returns:
        Monad: 결정화된 모나드
    """
    observer = get_llm_observer()
    crystallizer = get_llm_crystallizer()
    
    # 관측
    crystal = observer.observe(model_path)
    
    # 결정화
    monad = crystallizer.crystallize(crystal)
    
    return monad


# CLI
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python llm_crystallizer.py <path_to_model.safetensors>")
        sys.exit(1)
    
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    monad = digest_llm(sys.argv[1])
    
    print("\n" + "="*50)
    print(f"🧬 Monad Report")
    print("="*50)
    print(f"   Seed:     {monad.seed}")
    print(f"   Category: {monad.category.value}")
    print(f"   DNA:      {monad._dna}")
