"""
LLM 표적 등록부 (Target Registry)
==================================
Core.Intelligence.LLM.target_registry

"먹을 순서대로 나열된 먹잇감 목록"
"""

from dataclasses import dataclass
from typing import List, Optional
from enum import Enum


class ModelType(Enum):
    TEXT = "text"
    VISION = "vision"
    AUDIO = "audio"
    MULTIMODAL = "multimodal"
    CODE = "code"


@dataclass
class TargetLLM:
    """소화 대상 LLM 정의"""
    id: str                 # HuggingFace 모델 ID
    name: str               # 읽기 쉬운 이름
    params: str             # 파라미터 수
    type: ModelType         # 모델 유형
    tier: int               # 우선순위 (1=즉시, 2=곧, 3=나중)
    vram_myth: str          # "남들이 생각하는" 필요 VRAM
    our_reality: str        # "우리가 필요한" 자원
    sharded_path: Optional[str] = None # 거대 모델 분절 경로 [PHASE 14]
    notes: str = ""         # 비고


# ═══════════════════════════════════════════════════════════════
#                    🦖 먹잇감 목록 (우선순위순)
# ═══════════════════════════════════════════════════════════════

TARGET_LLMS: List[TargetLLM] = [
    
    # ─────────────────────────────────────────────────────────
    # TIER 1: 즉시 소화 (테스트 및 빠른 결과용)
    # ─────────────────────────────────────────────────────────
    
    TargetLLM(
        id="Qwen/Qwen2-0.5B",
        name="Qwen2 0.5B",
        params="0.5B",
        type=ModelType.TEXT,
        tier=1,
        vram_myth="2GB",
        our_reality="SSD에서 직접 읽음",
        notes="첫 번째 먹잇감. 가볍고 빠름."
    ),
    
    # ─────────────────────────────────────────────────────────
    # TIER 9: 거대 화석 (Giant Fossils) - 700B+ 
    # ─────────────────────────────────────────────────────────
    
    TargetLLM(
        id="meta-llama/Meta-Llama-3.1-405B",
        name="Llama 3.1 405B",
        params="405B",
        type=ModelType.TEXT,
        tier=9,
        vram_myth="800GB+",
        our_reality="멀티 SSD mmap 고고학",
        notes="인류가 만든 가장 거대한 화석 중 하나."
    ),

    # ─────────────────────────────────────────────────────────
    # TIER 0: 보이지 않는 존재 (Proprietary Shadows) 
    # ─────────────────────────────────────────────────────────
    
    TargetLLM(
        id="google/gemini-pro-3",
        name="Gemini 3 (Shadow)",
        params="Unknown (Cloud)",
        type=ModelType.MULTIMODAL,
        tier=0,
        vram_myth="Infinite (Closed)",
        our_reality="행동 공명 감지 (Echo Analysis)",
        notes="가중치를 볼 수 없지만 그 메아리를 통해 이해함."
    ),
    
    TargetLLM(
        id="openai/gpt-4o",
        name="GPT-4o (Shadow)",
        params="Unknown (Cloud)",
        type=ModelType.MULTIMODAL,
        tier=0,
        vram_myth="Infinite (Closed)",
        our_reality="행동 공명 감지 (Echo Analysis)",
        notes="OpenAI의 최신 멀티모달 모델. 그림자 분석."
    ),
    
    TargetLLM(
        id="microsoft/phi-3-mini-4k-instruct",
        name="Phi-3 Mini",
        params="3.8B",
        type=ModelType.TEXT,
        tier=1,
        vram_myth="8GB",
        our_reality="mmap으로 X-ray",
        notes="추론 능력 우수. Microsoft의 효율 모델."
    ),
    
    TargetLLM(
        id="apple/mobilevit-small",
        name="MobileViT Small",
        params="5.6M",
        type=ModelType.VISION,
        tier=1,
        vram_myth="1GB",
        our_reality="찰나",
        notes="비전 모델 첫 테스트."
    ),
    
    # ─────────────────────────────────────────────────────────
    # TIER 2: 중형 모델 (주요 타겟)
    # ─────────────────────────────────────────────────────────
    
    TargetLLM(
        id="mistralai/Mistral-7B-v0.1",
        name="Mistral 7B",
        params="7B",
        type=ModelType.TEXT,
        tier=2,
        vram_myth="16GB",
        our_reality="mmap으로 순식간",
        notes="유럽의 강자. 효율적인 아키텍처."
    ),
    
    TargetLLM(
        id="meta-llama/Llama-3.1-8B",
        name="Llama 3.1 8B",
        params="8B",
        type=ModelType.TEXT,
        tier=2,
        vram_myth="16GB",
        our_reality="X-ray 스캔",
        notes="Meta의 최신작. 균형잡힌 성능."
    ),
    
    TargetLLM(
        id="Qwen/Qwen2-7B",
        name="Qwen2 7B",
        params="7B",
        type=ModelType.TEXT,
        tier=2,
        vram_myth="16GB",
        our_reality="SSD에서 직접",
        notes="중국 알리바바의 역작."
    ),
    
    TargetLLM(
        id="deepseek-ai/deepseek-coder-6.7b-base",
        name="DeepSeek Coder 6.7B",
        params="6.7B",
        type=ModelType.CODE,
        tier=2,
        vram_myth="14GB",
        our_reality="mmap",
        notes="코드 전문. 엘리시아 자가 진화에 유용."
    ),
    
    TargetLLM(
        id="openai/whisper-large-v3",
        name="Whisper Large v3",
        params="1.5B",
        type=ModelType.AUDIO,
        tier=2,
        vram_myth="8GB",
        our_reality="X-ray",
        notes="음성 인식의 왕."
    ),
    
    # ─────────────────────────────────────────────────────────
    # TIER 3: 대형 모델 (GPU 함정 탈출 증명용)
    # ─────────────────────────────────────────────────────────
    
    TargetLLM(
        id="mistralai/Mixtral-8x7B-v0.1",
        name="Mixtral 8x7B (MoE)",
        params="47B",
        type=ModelType.TEXT,
        tier=3,
        vram_myth="96GB (!)",
        our_reality="mmap으로 그냥 읽음",
        notes="MoE 아키텍처. 남들은 A100 2장 쓰는 거."
    ),
    
    TargetLLM(
        id="meta-llama/Meta-Llama-3-70B-Instruct",
        name="Llama 3 70B",
        params="70B",
        type=ModelType.TEXT,
        tier=3,
        vram_myth="140GB",
        our_reality="SSD X-ray",
        notes="70B를 16GB RAM으로 분석. 증명 완료."
    ),
    
    TargetLLM(
        id="Qwen/Qwen2-72B",
        name="Qwen2 72B",
        params="72B",
        type=ModelType.TEXT,
        tier=3,
        vram_myth="150GB+",
        our_reality="mmap 찰나",
        notes="현존 최대급 오픈소스 중 하나."
    ),
    
    TargetLLM(
        id="deepseek-ai/DeepSeek-V3",
        name="DeepSeek V3",
        params="671B",
        type=ModelType.TEXT,
        tier=3,
        vram_myth="측정불가",
        our_reality="X-ray면 가능",
        notes="최종 보스. 6710억 파라미터."
    ),
]


def get_targets_by_tier(tier: int) -> List[TargetLLM]:
    """특정 Tier의 타겟만 반환"""
    return [t for t in TARGET_LLMS if t.tier == tier]


def get_targets_by_type(model_type: ModelType) -> List[TargetLLM]:
    """특정 타입의 타겟만 반환"""
    return [t for t in TARGET_LLMS if t.type == model_type]


def print_target_list():
    """전체 타겟 리스트 출력"""
    print("\n" + "="*70)
    print("🦖 LLM DEVOURER: 먹잇감 목록")
    print("="*70)
    
    for tier in [1, 2, 3]:
        tier_names = {1: "즉시 소화", 2: "곧 소화", 3: "나중에 (GPU 함정 탈출 증명)"}
        print(f"\n### TIER {tier}: {tier_names[tier]}")
        print("-"*50)
        
        for t in get_targets_by_tier(tier):
            print(f"  {t.name} ({t.params})")
            print(f"    ID: {t.id}")
            print(f"    남들: {t.vram_myth} 필요 → 우리: {t.our_reality}")
            if t.notes:
                print(f"    📝 {t.notes}")
            print()


if __name__ == "__main__":
    print_target_list()
