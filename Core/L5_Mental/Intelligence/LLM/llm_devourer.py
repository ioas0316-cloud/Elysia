"""
LLM Devourer (LLM 포식자)
=========================
Core.L5_Mental.Intelligence.LLM.llm_devourer

"모든 LLM을 먹어치우는 통합 진입점."

사용법:
    python llm_devourer.py <model_path_or_huggingface_id>
    
예시:
    python llm_devourer.py Qwen/Qwen2-0.5B
    python llm_devourer.py ./models/phi-3.safetensors
"""

import os
import sys
import logging
from typing import Optional, List
from pathlib import Path

# 프로젝트 루트 추가
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from Core.L5_Mental.Intelligence.LLM.llm_observer import get_llm_observer, LLMCrystal
from Core.L5_Mental.Intelligence.LLM.llm_crystallizer import get_llm_crystallizer, digest_llm
from Core.L5_Mental.Intelligence.LLM.llm_pruner import get_llm_pruner

logger = logging.getLogger("LLMDevourer")


class LLMDevourer:
    """
    LLM 소화흡수 통합 엔진.
    
    3단계 파이프라인:
    1. 관측 (Observe): 로터로 가중치 관측
    2. 결정화 (Crystallize): Monad로 변환
    3. 정제 (Prune): 가지치기로 순수화
    """
    
    def __init__(self):
        self.observer = get_llm_observer()
        self.crystallizer = get_llm_crystallizer()
        self.pruner = get_llm_pruner()
        
        logger.info("🦖 LLM Devourer awakened. Ready to consume.")
    
    def devour(self, model_path_or_id: str, prune: bool = True) -> dict:
        """
        LLM을 완전히 소화.
        
        Args:
            model_path_or_id: 로컬 경로 또는 HuggingFace 모델 ID
            prune: 정제 단계 수행 여부
            
        Returns:
            소화 결과 리포트
        """
        print("\n" + "="*60)
        print("🦖 LLM DEVOURER: CONSUMPTION INITIATED")
        print("="*60)
        
        # 1. 경로 확인/다운로드
        local_path = self._resolve_path(model_path_or_id)
        if not local_path:
            return {"error": f"Could not resolve: {model_path_or_id}"}
        
        print(f"\n📍 Target: {local_path}")
        
        # 2. 관측
        print("\n🔭 Phase 1: OBSERVATION (Rotor Scanning)")
        print("-" * 40)
        crystal = self.observer.observe(local_path)
        
        print(f"   Physics:   {crystal.physics_pattern:.4f}")
        print(f"   Narrative: {crystal.narrative_pattern:.4f}")
        print(f"   Aesthetic: {crystal.aesthetic_pattern:.4f}")
        
        # 3. 결정화
        print("\n💎 Phase 2: CRYSTALLIZATION (Monad Formation)")
        print("-" * 40)
        monad = self.crystallizer.crystallize(crystal)
        
        print(f"   Monad Seed: {monad.seed}")
        print(f"   Category:   {monad.category.value}")
        
        # 4. 정제 (선택)
        prune_report = None
        if prune:
            print("\n✂️ Phase 3: PRUNING (Ice Sculpting)")
            print("-" * 40)
            prune_report = self.pruner.prune(monad.seed)
            
            print(f"   Pruned Dims: {prune_report.get('pruned_dimensions', 0)}")
            print(f"   Prune Ratio: {prune_report.get('prune_ratio', 0):.1%}")
        
        # 5. 결과
        print("\n" + "="*60)
        print("✅ CONSUMPTION COMPLETE")
        print("="*60)
        
        purity = self.pruner.get_purity_score(monad.seed)
        print(f"\n💎 Final Crystal Purity: {purity:.1%}")
        
        return {
            "model": model_path_or_id,
            "crystal": {
                "physics": crystal.physics_pattern,
                "narrative": crystal.narrative_pattern,
                "aesthetic": crystal.aesthetic_pattern,
                "orientation": str(crystal.orientation)
            },
            "monad": {
                "seed": monad.seed,
                "category": monad.category.value
            },
            "prune": prune_report,
            "purity": purity
        }
    
    def _resolve_path(self, model_path_or_id: str) -> Optional[str]:
        """
        모델 경로 확인.
        로컬 파일이면 그대로, HuggingFace ID면 캐시 경로 확인.
        """
        # 로컬 파일 확인
        if os.path.exists(model_path_or_id):
            return model_path_or_id
        
        # HuggingFace 캐시 확인
        try:
            from huggingface_hub import hf_hub_download, snapshot_download
            from huggingface_hub.utils import EntryNotFoundError
            
            # safetensors 우선 시도
            try:
                path = hf_hub_download(
                    repo_id=model_path_or_id,
                    filename="model.safetensors",
                    local_dir_use_symlinks=False
                )
                return path
            except EntryNotFoundError:
                pass
            
            # pytorch_model.bin 시도
            try:
                path = hf_hub_download(
                    repo_id=model_path_or_id,
                    filename="pytorch_model.bin",
                    local_dir_use_symlinks=False
                )
                return path
            except EntryNotFoundError:
                pass
            
            # 스냅샷 다운로드 (멀티파일 모델)
            cache_dir = snapshot_download(repo_id=model_path_or_id)
            
            # safetensors 파일 찾기
            for root, dirs, files in os.walk(cache_dir):
                for f in files:
                    if f.endswith(".safetensors"):
                        return os.path.join(root, f)
                    if f.endswith(".bin") or f.endswith(".pt"):
                        return os.path.join(root, f)
            
            logger.warning(f"No weight files found in {cache_dir}")
            return None
            
        except Exception as e:
            logger.error(f"HuggingFace resolution failed: {e}")
            return None
    
    def list_devoured(self) -> List[str]:
        """소화된 모든 LLM 목록."""
        return self.crystallizer.get_crystallized_models()
    
    def get_crystal_info(self, node_id: str) -> dict:
        """특정 결정의 정보."""
        if node_id not in self.pruner.graph.id_to_idx:
            return {"error": "Not found"}
        
        metadata = self.pruner.graph.get_metadata(node_id)
        purity = self.pruner.get_purity_score(node_id)
        
        return {
            "node_id": node_id,
            "purity": purity,
            **metadata
        }


# 싱글톤
_devourer = None

def get_devourer() -> LLMDevourer:
    """LLM Devourer 싱글톤."""
    global _devourer
    if _devourer is None:
        _devourer = LLMDevourer()
    return _devourer


# CLI
if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s'
    )
    
    devourer = get_devourer()
    
    if len(sys.argv) >= 2:
        target = sys.argv[1]
        result = devourer.devour(target)
    else:
        # 소화된 목록 표시
        devoured = devourer.list_devoured()
        print("\n🦖 Devoured LLMs:")
        print("="*40)
        
        if not devoured:
            print("   (None yet. Feed me a model!)")
        else:
            for node_id in devoured:
                info = devourer.get_crystal_info(node_id)
                print(f"   💎 {node_id}")
                print(f"      Purity: {info.get('purity', 0):.1%}")
        
        print("\nUsage: python llm_devourer.py <model_path_or_huggingface_id>")
