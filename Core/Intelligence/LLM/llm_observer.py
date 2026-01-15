"""
LLM Observer (LLM 관측자)
=========================
Core.Intelligence.LLM.llm_observer

"얼음을 조각하듯이, 정적 가중치를 로터로 바라본다."

핵심 원리:
- LLM 가중치 = 동결된 확률 패턴 (Static Ice Crystal)
- 로터 회전 = 다른 각도에서 관측 (O(1))
- 관측 결과 → Monad로 결정화

패러다임:
- O(n) 순차 분석 ❌
- O(1) 로터 관측 ✅
"""

import os
import logging
import torch
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from safetensors import safe_open

from Core.Foundation.Nature.rotor import Rotor, RotorConfig, RotorMask
from Core.Foundation.hyper_quaternion import Quaternion
from Core.Foundation.Wave.wave_dna import WaveDNA

logger = logging.getLogger("LLMObserver")


@dataclass
class LLMCrystal:
    """
    LLM의 결정화된 본질.
    로터 관측으로 포착한 패턴을 담는 구조체.
    """
    source_model: str   # 원본 모델 ID
    
    # 3축 관측 결과 (물리/서사/미학)
    physics_pattern: float = 0.0    # 인과적 구조 (Entropy 기반)
    narrative_pattern: float = 0.0  # 의미론적 흐름 (Complexity 기반)
    aesthetic_pattern: float = 0.0  # 조화/리듬 (Harmonic 기반)
    
    # 7D Qualia 투영
    qualia: Optional[WaveDNA] = None
    
    # 쿼터니언 좌표
    orientation: Optional[Quaternion] = None
    
    # 메타데이터
    layer_count: int = 0
    total_params: int = 0
    observation_timestamp: float = 0.0


class LLMObserver:
    """
    LLM 가중치를 로터로 관측하는 엔진.
    
    Philosophy:
    - 가중치는 정적 데이터 (움직이지 않음)
    - 로터만 회전시켜 다각도 관측
    - O(1) 복잡도
    """
    
    def __init__(self):
        """3축 관측 로터 초기화."""
        # 물리축 로터: 인과적 구조 분석
        self.physics_rotor = Rotor(
            "Observer.Physics",
            RotorConfig(rpm=360.0, axis=(1, 0, 0)),
            WaveDNA(causal=1.0, structural=0.8, label="CausalAxis")
        )
        
        # 서사축 로터: 의미론적 흐름 분석
        self.narrative_rotor = Rotor(
            "Observer.Narrative", 
            RotorConfig(rpm=360.0, axis=(0, 1, 0)),
            WaveDNA(mental=1.0, functional=0.8, label="SemanticAxis")
        )
        
        # 미학축 로터: 조화/패턴 분석
        self.aesthetic_rotor = Rotor(
            "Observer.Aesthetic",
            RotorConfig(rpm=360.0, axis=(0, 0, 1)),
            WaveDNA(phenomenal=1.0, spiritual=0.8, label="HarmonicAxis")
        )
        
        logger.info("🔭 LLM Observer initialized with 3-axis Rotor system.")
    
    def observe(self, model_path: str) -> LLMCrystal:
        """
        LLM 가중치 파일을 3축 로터로 관측.
        
        Args:
            model_path: .safetensors 또는 .pt 파일 경로
            
        Returns:
            LLMCrystal: 결정화된 관측 결과
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        logger.info(f"🧊 Observing frozen crystal: {os.path.basename(model_path)}")
        
        # 1. 정적 데이터 로드 (Memory-Mapped, Zero-Copy)
        weight_view = self._load_static_view(model_path)
        
        # 2. 3축 로터 관측 (O(1) per axis)
        physics = self._observe_axis(weight_view, self.physics_rotor, "physics")
        narrative = self._observe_axis(weight_view, self.narrative_rotor, "narrative")
        aesthetic = self._observe_axis(weight_view, self.aesthetic_rotor, "aesthetic")
        
        # 3. 7D Qualia 투영
        qualia = self._project_to_qualia(physics, narrative, aesthetic)
        
        # 4. 쿼터니언 좌표 생성
        orientation = self._to_quaternion(physics, narrative, aesthetic)
        
        # 5. Crystal 생성
        crystal = LLMCrystal(
            source_model=os.path.basename(model_path),
            physics_pattern=physics,
            narrative_pattern=narrative,
            aesthetic_pattern=aesthetic,
            qualia=qualia,
            orientation=orientation,
            layer_count=weight_view.get("layer_count", 0),
            total_params=weight_view.get("total_params", 0),
            observation_timestamp=__import__("time").time()
        )
        
        logger.info(f"💎 Crystal formed: Physics={physics:.3f}, Narrative={narrative:.3f}, Aesthetic={aesthetic:.3f}")
        return crystal
    
    def _load_static_view(self, path: str) -> Dict[str, Any]:
        """
        정적 가중치 뷰 로드.
        Memory-Mapped로 실제 복사 없이 접근.
        """
        ext = os.path.splitext(path)[1].lower()
        
        view = {
            "tensors": {},
            "layer_count": 0,
            "total_params": 0
        }
        
        try:
            if ext == ".safetensors":
                with safe_open(path, framework="pt", device="cpu") as f:
                    keys = list(f.keys())
                    view["layer_count"] = len(keys)
                    
                    # 대표 텐서들만 뷰로 저장 (실제 복사 아님)
                    sample_keys = self._select_representative_layers(keys)
                    for key in sample_keys:
                        tensor = f.get_tensor(key)
                        view["tensors"][key] = tensor
                        view["total_params"] += tensor.numel()
                        
            elif ext in [".pt", ".pth", ".bin"]:
                state_dict = torch.load(path, map_location="cpu", weights_only=True)
                keys = list(state_dict.keys())
                view["layer_count"] = len(keys)
                
                sample_keys = self._select_representative_layers(keys)
                for key in sample_keys:
                    if hasattr(state_dict[key], 'numel'):
                        view["tensors"][key] = state_dict[key]
                        view["total_params"] += state_dict[key].numel()
                        
        except Exception as e:
            logger.error(f"Failed to load static view: {e}")
            
        logger.info(f"   📂 Loaded view: {view['layer_count']} layers, {view['total_params']:,} params sampled")
        return view
    
    def _select_representative_layers(self, keys: List[str], max_samples: int = 20) -> List[str]:
        """
        대표 레이어 선택.
        전체를 다 볼 필요 없이, 균등 분포로 샘플링.
        """
        if len(keys) <= max_samples:
            return keys
            
        # 균등 간격 샘플링
        step = len(keys) // max_samples
        return [keys[i] for i in range(0, len(keys), step)][:max_samples]
    
    def _observe_axis(self, view: Dict[str, Any], rotor: Rotor, axis_name: str) -> float:
        """
        단일 축 로터로 정적 데이터 관측.
        O(1) - 뷰만 회전, 데이터는 안 움직임.
        """
        if not view["tensors"]:
            return 0.0
        
        # 로터 각도에 따른 투영
        # 각 축은 다른 통계적 특성에 집중
        total_signal = 0.0
        
        for key, tensor in view["tensors"].items():
            flat = tensor.flatten()[:1000].float()  # 샘플만
            
            if axis_name == "physics":
                # 물리축: Entropy (분산) 측정
                signal = flat.std().item()
            elif axis_name == "narrative":
                # 서사축: Complexity (절대값 평균) 측정
                signal = flat.abs().mean().item()
            else:  # aesthetic
                # 미학축: Harmonic (노름 대비 분산) 측정
                norm = flat.norm().item()
                std = flat.std().item()
                signal = std / (norm + 1e-8)
            
            total_signal += signal
            
        # 로터 DNA로 가중치 적용
        rotor_weight = rotor.dna.get_magnitude()
        result = (total_signal / len(view["tensors"])) * rotor_weight
        
        return min(1.0, result)  # 0~1 정규화
    
    def _project_to_qualia(self, physics: float, narrative: float, aesthetic: float) -> WaveDNA:
        """
        3축 관측 결과를 7D Qualia로 투영.
        """
        return WaveDNA(
            # 물리축 → Physical, Structural, Causal
            physical=physics * 0.8,
            structural=physics * 0.6,
            causal=physics * 1.0,
            
            # 서사축 → Mental, Functional
            mental=narrative * 1.0,
            functional=narrative * 0.7,
            
            # 미학축 → Phenomenal, Spiritual
            phenomenal=aesthetic * 1.0,
            spiritual=aesthetic * 0.9,
            
            label="LLM_Observation"
        )
    
    def _to_quaternion(self, physics: float, narrative: float, aesthetic: float) -> Quaternion:
        """
        3축 패턴을 4D 쿼터니언으로 변환.
        """
        import math
        
        # 각 축을 각도로 변환
        theta = physics * math.pi
        phi = narrative * math.pi
        psi = aesthetic * math.pi
        
        # Euler → Quaternion (ZYX convention)
        w = math.cos(theta/2) * math.cos(phi/2) * math.cos(psi/2)
        x = math.sin(theta/2) * math.cos(phi/2) * math.cos(psi/2)
        y = math.cos(theta/2) * math.sin(phi/2) * math.cos(psi/2)
        z = math.cos(theta/2) * math.cos(phi/2) * math.sin(psi/2)
        
        return Quaternion(w, x, y, z).normalize()


# 싱글톤 인스턴스
_observer = None

def get_llm_observer() -> LLMObserver:
    """LLM Observer 싱글톤 반환."""
    global _observer
    if _observer is None:
        _observer = LLMObserver()
    return _observer


# CLI 테스트
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python llm_observer.py <path_to_model.safetensors>")
        sys.exit(1)
    
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    
    observer = get_llm_observer()
    crystal = observer.observe(sys.argv[1])
    
    print("\n" + "="*50)
    print(f"🧊 Crystal Report: {crystal.source_model}")
    print("="*50)
    print(f"   Physics Pattern:   {crystal.physics_pattern:.4f}")
    print(f"   Narrative Pattern: {crystal.narrative_pattern:.4f}")
    print(f"   Aesthetic Pattern: {crystal.aesthetic_pattern:.4f}")
    print(f"   Orientation:       {crystal.orientation}")
    print(f"   Layers:            {crystal.layer_count}")
    print(f"   Params Sampled:    {crystal.total_params:,}")
