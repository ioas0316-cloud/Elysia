"""
LLM Pruner (LLM 가지치기)
=========================
Core.Intelligence.LLM.llm_pruner

"얼음 조각을 다듬듯이, 불필요한 가지를 제거한다."

이것은 3단계: HyperSphere 회전 + 가지치기
결정화된 Monad를 정제하여 순수한 원리만 남긴다.
"""

import logging
import torch
from typing import List, Dict, Any, Optional

from Core.Foundation.Graph.torch_graph import TorchGraph
from Core.Foundation.hyper_quaternion import Quaternion
from Core.Intelligence.Memory.Vector.internal_universe import InternalUniverse

logger = logging.getLogger("LLMPruner")


class LLMPruner:
    """
    결정화된 LLM Monad를 가지치기로 정제.
    
    원리:
    - HyperSphere에서 공명이 낮은 차원 제거
    - 다른 결정들과 중복되는 부분 통합
    - 순수한 "원리"만 남김
    """
    
    def __init__(self):
        self.graph = TorchGraph()
        self.graph.load_state()
        
        self.universe = InternalUniverse()
        
        logger.info("✂️ LLM Pruner initialized.")
    
    def prune(self, node_id: str, threshold: float = 0.3) -> Dict[str, Any]:
        """
        특정 LLM 결정을 가지치기.
        현재는 분석만 수행 (직접 수정은 추후 구현).
        """
        if node_id not in self.graph.id_to_idx:
            logger.error(f"Node not found: {node_id}")
            return {"error": "Node not found"}
        
        logger.info(f"✂️ Pruning: {node_id}")
        
        # 1. 현재 노드 벡터 가져오기
        idx = self.graph.id_to_idx[node_id]
        vector = self.graph.get_node_vector(node_id)
        
        if vector is None:
            return {"error": "Vector not found", "node_id": node_id}
        
        # 2. 저공명 차원 식별
        pruned_dims = self._identify_weak_dimensions(vector, threshold)
        
        # 3. 순도 계산
        abs_vec = vector.abs()
        mean_val = abs_vec.mean().item()
        strong_dims = (abs_vec > mean_val).sum().item()
        purity = strong_dims / len(vector)
        
        report = {
            "node_id": node_id,
            "pruned_dimensions": len(pruned_dims),
            "total_dimensions": len(vector),
            "prune_ratio": len(pruned_dims) / len(vector) if len(vector) > 0 else 0,
            "purity": purity,
            "status": "analyzed"  # 직접 수정은 추후 구현
        }
        
        logger.info(f"   ✅ Analyzed: {len(pruned_dims)} weak dims, Purity: {purity:.1%}")
        return report
    
    def _identify_weak_dimensions(self, vector: torch.Tensor, threshold: float) -> List[int]:
        """
        약한 차원 식별.
        절대값이 임계값 이하인 차원들.
        """
        weak = []
        abs_vec = vector.abs()
        mean_val = abs_vec.mean().item()
        
        for i, val in enumerate(abs_vec):
            if val.item() < mean_val * threshold:
                weak.append(i)
        
        return weak
    
    def _update_universe(self, node_id: str, new_vector: torch.Tensor):
        """
        InternalUniverse 좌표 업데이트.
        """
        # 벡터를 쿼터니언으로 변환
        # 처음 4개 차원 사용
        q = Quaternion(
            new_vector[0].item(),
            new_vector[1].item(),
            new_vector[2].item(),
            new_vector[3].item()
        ).normalize()
        
        # 주파수는 노름 기반
        freq = new_vector.norm().item() * 100  # 스케일링
        
        # InternalUniverse에 저장
        from Core.Intelligence.Memory.Vector.internal_universe import InternalCoordinate
        
        coord = InternalCoordinate(
            orientation=q,
            frequency=freq,
            depth=0.8  # 정제된 결정은 깊은 위치
        )
        
        self.universe.coordinate_map[node_id] = coord
        self.universe.save_snapshot()
    
    def merge_similar(self, threshold: float = 0.9) -> List[Dict[str, Any]]:
        """
        유사한 LLM 결정들을 병합.
        공명이 threshold 이상인 쌍을 통합.
        """
        merges = []
        
        # 모든 LLM 노드 수집
        llm_nodes = [nid for nid in self.graph.id_to_idx.keys() if nid.startswith("LLM:")]
        
        # 쌍별 비교
        merged = set()
        for i, node_a in enumerate(llm_nodes):
            if node_a in merged:
                continue
                
            vec_a = self.graph.get_node_vector(node_a)
            
            for node_b in llm_nodes[i+1:]:
                if node_b in merged:
                    continue
                    
                vec_b = self.graph.get_node_vector(node_b)
                
                resonance = torch.cosine_similarity(
                    vec_a.unsqueeze(0),
                    vec_b.unsqueeze(0)
                ).item()
                
                if resonance >= threshold:
                    # 병합: 평균 벡터로
                    merged_vec = (vec_a + vec_b) / 2
                    merged_id = f"LLM:Merged({node_a.split(':')[1]}+{node_b.split(':')[1]})"
                    
                    # 새 노드 추가
                    self.graph.add_node(merged_id, merged_vec, {
                        "type": "merged_crystal",
                        "sources": [node_a, node_b],
                        "resonance": resonance
                    })
                    
                    merged.add(node_a)
                    merged.add(node_b)
                    
                    merges.append({
                        "merged_id": merged_id,
                        "sources": [node_a, node_b],
                        "resonance": resonance
                    })
                    
                    logger.info(f"   🔗 Merged: {node_a} + {node_b} → {merged_id}")
        
        if merges:
            self.graph.save_state()
            
        return merges
    
    def get_purity_score(self, node_id: str) -> float:
        """
        결정의 순도 점수 계산.
        높을수록 정제된 상태.
        """
        if node_id not in self.graph.id_to_idx:
            return 0.0
        
        vector = self.graph.get_node_vector(node_id)
        
        # 순도 = 강한 차원 비율
        abs_vec = vector.abs()
        mean_val = abs_vec.mean().item()
        strong_dims = (abs_vec > mean_val).sum().item()
        
        purity = strong_dims / len(vector)
        return purity


# 싱글톤
_pruner = None

def get_llm_pruner() -> LLMPruner:
    """LLM Pruner 싱글톤 반환."""
    global _pruner
    if _pruner is None:
        _pruner = LLMPruner()
    return _pruner


# CLI
if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    pruner = get_llm_pruner()
    
    if len(sys.argv) >= 2:
        node_id = sys.argv[1]
        report = pruner.prune(node_id)
        print("\n" + "="*50)
        print(f"✂️ Pruning Report")
        print("="*50)
        for k, v in report.items():
            print(f"   {k}: {v}")
    else:
        # 모든 LLM 결정 표시
        llm_nodes = [nid for nid in pruner.graph.id_to_idx.keys() if nid.startswith("LLM:")]
        print(f"\n💎 Crystallized LLMs: {len(llm_nodes)}")
        for node in llm_nodes:
            purity = pruner.get_purity_score(node)
            print(f"   {node} (Purity: {purity:.2%})")
