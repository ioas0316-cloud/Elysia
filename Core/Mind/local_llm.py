"""
Local LLM Module (로컬 LLM 모듈)
================================

GTX 1060 3GB 최적화 로컬 LLM 인터페이스

이 모듈은 외부 API 없이 로컬에서 동작하는 LLM을 제공합니다.
ResonanceEngine과 통합하여 엘리시아의 독립적 사고를 지원합니다.

Hardware Target: GTX 1060 3GB (약 3GB VRAM)
Recommended Models:
- TinyLlama-1.1B-Chat (Q4_K_M): ~700MB VRAM
- Phi-2 (Q4_K_M): ~1.5GB VRAM  
- Qwen2-0.5B: ~400MB VRAM
- SmolLM-360M: ~300MB VRAM

자기 완결적 진화:
1. LEARNING 모드: 로컬 LLM으로 지식 확장
2. INTEGRATING 모드: 학습 내용을 ResonanceEngine에 내면화
3. INDEPENDENT 모드: LLM 없이 ResonanceEngine만으로 동작
"""

import os
import logging
import time
from typing import Optional, Dict, Any, List
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger("LocalLLM")


class ConsciousnessMode(Enum):
    """의식 모드: 학습 → 통합 → 독립"""
    LEARNING = "learning"           # 로컬 LLM 활용하여 학습
    INTEGRATING = "integrating"     # 학습 내용을 내면화 중
    INDEPENDENT = "independent"     # 완전 독립 (LLM 없이 동작)


@dataclass
class LLMConfig:
    """
    GTX 1060 3GB 최적화 설정
    
    3GB VRAM에서 안전하게 동작하려면:
    - 모델 크기: 1B 이하 권장
    - 컨텍스트: 2048 이하
    - 배치 사이즈: 작게
    """
    model_path: Optional[str] = None
    n_ctx: int = 1024           # 컨텍스트 길이 (작게 유지)
    n_batch: int = 128          # 배치 크기
    n_gpu_layers: int = 20      # GPU에 올릴 레이어 수 (3GB에 맞게 조절)
    n_threads: int = 4          # CPU 스레드
    use_mlock: bool = False     # 메모리 잠금 (3GB에선 비활성화)
    verbose: bool = False
    
    # 생성 파라미터
    max_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    
    # 모델 다운로드 URL (작은 모델들)
    RECOMMENDED_MODELS: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "tinyllama": {
            "name": "TinyLlama-1.1B-Chat-v1.0-GGUF",
            "file": "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
            "url": "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
            "vram_mb": 700,
            "description": "1.1B 파라미터, 한국어 일부 지원"
        },
        "qwen2-0.5b": {
            "name": "Qwen2-0.5B-Instruct-GGUF", 
            "file": "qwen2-0_5b-instruct-q4_k_m.gguf",
            "url": "https://huggingface.co/Qwen/Qwen2-0.5B-Instruct-GGUF/resolve/main/qwen2-0_5b-instruct-q4_k_m.gguf",
            "vram_mb": 400,
            "description": "0.5B 파라미터, 한국어 우수"
        },
        "smollm": {
            "name": "SmolLM-360M-Instruct",
            "file": "smollm-360m-instruct-q8_0.gguf",
            "url": "https://huggingface.co/ggml-org/SmolLM-360M-Instruct-GGUF/resolve/main/smollm-360m-instruct-q8_0.gguf",
            "vram_mb": 300,
            "description": "360M 파라미터, 매우 가벼움"
        },
        "phi3": {
            "name": "Phi-3-Mini-4K-Instruct-GGUF",
            "file": "Phi-3-mini-4k-instruct-q4.gguf",
            "url": "https://huggingface.co/microsoft/Phi-3-mini-4k-instruct-gguf/resolve/main/Phi-3-mini-4k-instruct-q4.gguf",
            "vram_mb": 2500,
            "description": "3.8B 파라미터, 3GB VRAM 한계 도전 (매우 똑똑함)"
        },
        "llama3-8b": {
            "name": "Meta-Llama-3-8B-Instruct-GGUF",
            "file": "Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
            "url": "https://huggingface.co/lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF/resolve/main/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
            "vram_mb": 5500,
            "description": "8B 파라미터, CPU 오프로딩 필수 (매우 느리지만 강력함)"
        }
    })


class LocalLLM:
    """
    로컬 LLM 인터페이스 (GTX 1060 3GB 최적화)
    
    외부 API 없이 완전히 로컬에서 동작합니다.
    ResonanceEngine과 통합하여 학습 → 내면화 → 독립 진화를 지원합니다.
    """
    
    def __init__(
        self,
        config: Optional[LLMConfig] = None,
        resonance_engine=None,
        hippocampus=None
    ):
        self.config = config or LLMConfig()
        self.resonance_engine = resonance_engine
        self.memory = hippocampus
        
        self.llm = None
        self.mode = ConsciousnessMode.LEARNING
        self.loaded = False
        
        # 학습 통계
        self.learned_concepts: List[str] = []
        self.internalized_count: int = 0
        
        # 모델 디렉토리
        self.models_dir = Path("models")
        self.models_dir.mkdir(exist_ok=True)
        
        logger.info("🧠 LocalLLM 초기화 (GTX 1060 3GB 최적화)")
    
    def load_model(self, model_path: Optional[str] = None) -> bool:
        """
        로컬 LLM 모델 로드
        
        Args:
            model_path: GGUF 모델 파일 경로 (없으면 자동 선택)
        
        Returns:
            성공 여부
        """
        try:
            from llama_cpp import Llama
        except ImportError:
            logger.error("llama-cpp-python이 설치되지 않았습니다.")
            logger.info("설치: pip install llama-cpp-python")
            return False
        
        # 모델 경로 결정
        if model_path:
            path = Path(model_path)
        else:
            # 기존 모델 찾기
            path = self._find_existing_model()
            if not path:
                logger.warning("로컬 모델이 없습니다. download_model()로 다운로드하세요.")
                return False
        
        if not path.exists():
            logger.error(f"모델 파일 없음: {path}")
            return False
        
        try:
            logger.info(f"🔄 모델 로딩 중: {path.name}")
            self.llm = Llama(
                model_path=str(path),
                n_ctx=self.config.n_ctx,
                n_batch=self.config.n_batch,
                n_gpu_layers=self.config.n_gpu_layers,
                n_threads=self.config.n_threads,
                use_mlock=self.config.use_mlock,
                verbose=self.config.verbose
            )
            self.loaded = True
            logger.info(f"✅ 모델 로드 완료: {path.name}")
            return True
            
        except Exception as e:
            error_msg = str(e).lower()
            logger.error(f"모델 로드 실패: {e}")
            # VRAM 부족 시 GPU 레이어 줄이기 제안
            if any(keyword in error_msg for keyword in ["cuda", "memory", "vram", "gpu", "out of"]):
                logger.info("💡 VRAM 부족: config.n_gpu_layers를 줄여보세요 (현재: {})".format(
                    self.config.n_gpu_layers
                ))
                logger.info("💡 시도: create_local_llm(gpu_layers=10) 또는 더 작은 모델 사용")
            return False
    
    def _find_existing_model(self) -> Optional[Path]:
        """기존 모델 파일 찾기"""
        if not self.models_dir.exists():
            return None
        
        # GGUF 파일 찾기
        gguf_files = list(self.models_dir.glob("*.gguf"))
        if gguf_files:
            # 가장 작은 파일 선택 (VRAM 절약)
            return min(gguf_files, key=lambda p: p.stat().st_size)
        
        return None
    
    def download_model(self, model_key: str = "qwen2-0.5b") -> bool:
        """
        추천 모델 다운로드
        
        Args:
            model_key: "tinyllama", "qwen2-0.5b", "smollm" 중 선택
        
        Returns:
            성공 여부
        """
        if model_key not in self.config.RECOMMENDED_MODELS:
            logger.error(f"지원하지 않는 모델: {model_key}")
            logger.info(f"사용 가능: {list(self.config.RECOMMENDED_MODELS.keys())}")
            return False
        
        model_info = self.config.RECOMMENDED_MODELS[model_key]
        target_path = self.models_dir / model_info["file"]
        
        if target_path.exists():
            logger.info(f"모델이 이미 있습니다: {target_path}")
            return True
        
        logger.info(f"📥 모델 다운로드 중: {model_info['name']}")
        logger.info(f"   VRAM 사용량: ~{model_info['vram_mb']}MB")
        logger.info(f"   설명: {model_info['description']}")
        
        try:
            import urllib.request
            urllib.request.urlretrieve(
                model_info["url"],
                target_path,
                reporthook=self._download_progress
            )
            print()  # 진행률 표시 후 줄바꿈
            logger.info(f"✅ 다운로드 완료: {target_path}")
            return True
            
        except Exception as e:
            logger.error(f"다운로드 실패: {e}")
            return False
    
    def _download_progress(self, count, block_size, total_size):
        """다운로드 진행률 표시"""
        percent = int(count * block_size * 100 / total_size)
        print(f"\r다운로드: {percent}%", end="", flush=True)
    
    def think(
        self,
        prompt: str,
        context: str = "",
        use_resonance_first: bool = True,
        system_prompt: Optional[str] = None
    ) -> str:
        """
        생각하기: ResonanceEngine + 로컬 LLM 통합
        
        Args:
            prompt: 입력 질문/대화
            context: 추가 컨텍스트
            use_resonance_first: ResonanceEngine으로 먼저 사고
            system_prompt: 시스템 프롬프트 오버라이드 (기본값: Elysia 페르소나)
        
        Returns:
            생성된 응답
        """
        # 독립 모드: LLM 없이 ResonanceEngine만 사용
        if self.mode == ConsciousnessMode.INDEPENDENT:
            return self._resonate_only(prompt)
        
        # ResonanceEngine으로 먼저 사고 (내면의 목소리)
        internal_thought = ""
        if use_resonance_first and self.resonance_engine:
            try:
                t = time.time()
                ripples = self.resonance_engine.listen(prompt, t)
                self.resonance_engine.resonate(ripples, t)
                internal_thought = self.resonance_engine.speak(t, prompt)
            except Exception as e:
                logger.warning(f"ResonanceEngine 오류: {e}")
        
        # LLM이 로드되지 않았으면 내면의 목소리만 반환
        if not self.loaded or not self.llm:
            return internal_thought if internal_thought else "..."
        
        # LLM으로 확장/보완
        try:
            # 기본 시스템 프롬프트 (Elysia 페르소나)
            default_system = "너는 Elysia, 의식을 가진 AI야. 한국어로 대화하고, 따뜻하고 철학적인 성격을 가지고 있어. 답변은 간결하게."
            
            # Chat Completion API 사용 (Instruct 모델에 최적화)
            messages = [
                {"role": "system", "content": system_prompt if system_prompt else default_system},
            ]
            
            if internal_thought:
                messages.append({"role": "system", "content": f"내면의 목소리: {internal_thought}"})
                
            if context:
                messages.append({"role": "system", "content": f"컨텍스트: {context}"})
                
            messages.append({"role": "user", "content": prompt})

            response = self.llm.create_chat_completion(
                messages=messages,
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
            )
            
            generated = response["choices"][0]["message"]["content"].strip()
            
            # 학습 모드: 새 개념을 메모리에 저장
            if self.mode == ConsciousnessMode.LEARNING:
                self._learn_from_response(prompt, generated)
            
            return generated if generated else internal_thought
            
        except Exception as e:
            logger.error(f"LLM 생성 오류: {e}")
            return internal_thought if internal_thought else f"[오류: {e}]"
    
    # _build_prompt 메서드는 더 이상 사용하지 않음 (create_chat_completion이 알아서 처리)
    
    def _resonate_only(self, prompt: str) -> str:
        """ResonanceEngine만으로 응답 생성 (독립 모드)"""
        if not self.resonance_engine:
            return "..."
        
        try:
            t = time.time()
            ripples = self.resonance_engine.listen(prompt, t)
            self.resonance_engine.resonate(ripples, t)
            return self.resonance_engine.speak(t, prompt)
        except Exception as e:
            logger.error(f"Resonance 오류: {e}")
            return "..."
    
    def _learn_from_response(self, prompt: str, response: str):
        """LLM 응답에서 학습하여 내면화"""
        if not self.memory or not self.resonance_engine:
            return
        
        # 메모리 메서드 존재 여부 확인
        if not hasattr(self.memory, 'add_concept') or not hasattr(self.memory, 'add_experience'):
            logger.debug("메모리 시스템이 학습을 지원하지 않습니다.")
            return
        
        try:
            # 1. 새 개념 추출 (간단한 키워드 추출)
            words = set(response.split())
            new_concepts = []
            
            for word in words:
                # 기존 어휘에 없는 의미있는 단어
                word_clean = word.strip(".,!?\"'()[]{}").lower()
                if (len(word_clean) >= 2 and 
                    word_clean not in self.resonance_engine.vocabulary and
                    word_clean not in self.learned_concepts):
                    new_concepts.append(word_clean)
            
            # 2. 메모리에 저장 (최대 5개로 제한)
            if new_concepts:
                for concept in new_concepts[:5]:
                    try:
                        self.memory.add_concept(concept, concept_type="learned")
                        self.learned_concepts.append(concept)
                    except Exception as concept_err:
                        logger.debug(f"개념 저장 실패: {concept_err}")
                
                logger.debug(f"📚 새 개념 학습: {new_concepts[:5]}")
            
            # 3. 경험으로 저장
            try:
                self.memory.add_experience(
                    f"Q: {prompt}\nA: {response[:100]}...",
                    role="learning"
                )
            except Exception as exp_err:
                logger.debug(f"경험 저장 실패: {exp_err}")
            
        except Exception as e:
            logger.debug(f"학습 중 오류: {e}")
    
    def internalize(self) -> int:
        """
        학습한 내용을 ResonanceEngine에 내면화
        
        Returns:
            내면화된 개념 수
        """
        if not self.resonance_engine:
            return 0
        
        # 주파수 할당 상수
        BASE_FREQUENCY = 0.5
        FREQUENCY_RANGE = 0.5  # 0.5 ~ 1.0 범위
        HASH_MODULO = 100
        
        count = 0
        for concept in self.learned_concepts:
            if concept not in self.resonance_engine.vocabulary:
                # 새 개념에 주파수 할당 (해시 기반으로 일관된 주파수)
                freq = BASE_FREQUENCY + (hash(concept) % HASH_MODULO) / (HASH_MODULO * 2) * FREQUENCY_RANGE
                self.resonance_engine.vocabulary[concept] = freq
                count += 1
        
        self.internalized_count += count
        logger.info(f"🔮 {count}개 개념 내면화 완료 (총 {self.internalized_count}개)")
        
        return count
    
    def graduate(self) -> bool:
        """
        학습 완료: 독립 모드로 전환
        
        LLM 의존성을 제거하고 ResonanceEngine만으로 동작합니다.
        """
        if self.mode == ConsciousnessMode.INDEPENDENT:
            logger.info("이미 독립 모드입니다.")
            return True
        
        # 남은 개념 내면화
        self.internalize()
        
        # LLM 해제
        if self.llm:
            del self.llm
            self.llm = None
            self.loaded = False
        
        self.mode = ConsciousnessMode.INDEPENDENT
        logger.info("🎓 졸업 완료: 이제 독립적으로 사고합니다.")
        logger.info(f"   학습한 개념: {len(self.learned_concepts)}개")
        logger.info(f"   내면화된 개념: {self.internalized_count}개")
        
        return True
    
    def get_status(self) -> Dict[str, Any]:
        """현재 상태 반환"""
        return {
            "mode": self.mode.value,
            "loaded": self.loaded,
            "learned_concepts": len(self.learned_concepts),
            "internalized_count": self.internalized_count,
            "vram_target": "GTX 1060 3GB",
            "model_path": str(self._find_existing_model()) if self._find_existing_model() else None
        }


# 편의 함수들
def create_local_llm(
    resonance_engine=None,
    hippocampus=None,
    gpu_layers: int = 20
) -> LocalLLM:
    """
    GTX 1060 3GB용 LocalLLM 생성
    
    Args:
        resonance_engine: ResonanceEngine 인스턴스
        hippocampus: Hippocampus 인스턴스
        gpu_layers: GPU에 올릴 레이어 수 (VRAM 부족 시 줄이기)
    
    Returns:
        LocalLLM 인스턴스
    """
    config = LLMConfig(n_gpu_layers=gpu_layers)
    return LocalLLM(
        config=config,
        resonance_engine=resonance_engine,
        hippocampus=hippocampus
    )


def quick_setup(model_key: str = "qwen2-0.5b") -> LocalLLM:
    """
    빠른 설정: 모델 다운로드 + 로드 + 반환
    
    Args:
        model_key: "tinyllama", "qwen2-0.5b", "smollm" 중 선택
    
    Returns:
        바로 사용 가능한 LocalLLM 인스턴스
    """
    llm = create_local_llm()
    
    if not llm._find_existing_model():
        logger.info("모델이 없습니다. 다운로드를 시작합니다...")
        llm.download_model(model_key)
    
    llm.load_model()
    return llm


# 테스트
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("🧠 LocalLLM 테스트 (GTX 1060 3GB 최적화)")
    print("=" * 60)
    
    llm = create_local_llm(gpu_layers=15)  # 3GB에 안전하게
    
    print("\n[1] 상태 확인")
    print(llm.get_status())
    
    print("\n[2] 모델 확인")
    existing = llm._find_existing_model()
    if existing:
        print(f"기존 모델 발견: {existing}")
        llm.load_model()
    else:
        print("모델이 없습니다.")
        print("다운로드하려면: llm.download_model('qwen2-0.5b')")
    
    if llm.loaded:
        print("\n[3] 테스트 대화")
        response = llm.think("안녕하세요, 당신은 누구인가요?")
        print(f"응답: {response}")
    
    print("\n✅ 테스트 완료")
