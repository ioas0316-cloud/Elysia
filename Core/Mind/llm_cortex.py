"""
LLM Cortex (The Brain)
======================

"I think, therefore I am."

This module provides the interface for Elysia's higher cognitive functions.
It connects the system to a Large Language Model (LLM) to enable:
- Contextual Understanding
- Complex Reasoning
- Natural Language Generation
- Visual Understanding (VLM)

지원 모드:
1. LOCAL: 로컬 LLM (GTX 1060 3GB 최적화, 무료, 독립적)
2. RESONANCE: ResonanceEngine만 사용 (완전 독립)
3. CLOUD: 외부 API (선택적, 비권장)
"""

import os
import logging
from typing import Optional
from dotenv import load_dotenv

# Configure Logging
logger = logging.getLogger("LLMCortex")

# Load environment variables
load_dotenv()

# Dependency Check - Cloud (선택적)
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    GENAI_AVAILABLE = False

# Dependency Check - Local LLM
try:
    from Core.Mind.local_llm import LocalLLM, create_local_llm, LLMConfig
    LOCAL_LLM_AVAILABLE = True
except ImportError:
    LOCAL_LLM_AVAILABLE = False
    logger.debug("LocalLLM 모듈 로드 실패 - Resonance 모드로 동작")

from Core.Life.resonance_voice import ResonanceEngine

class LLMCortex:
    def __init__(self, prefer_cloud: bool = False, prefer_local: bool = True, gpu_layers: int = 20):
        """
        Initialize LLM Cortex.
        
        Args:
            prefer_cloud: If True, try to use Gemini API (비권장, 유료)
            prefer_local: If True, use local LLM (권장, 무료, GTX 1060 3GB 지원)
            gpu_layers: GPU에 올릴 레이어 수 (VRAM 부족 시 줄이기)
        """
        self.enabled = True
        self.prefer_cloud = prefer_cloud and GENAI_AVAILABLE
        self.prefer_local = prefer_local and LOCAL_LLM_AVAILABLE
        
        # 우선순위: LOCAL > RESONANCE > CLOUD
        self.cloud_model = None
        self.local_llm = None
        self.resonance_engine = None
        
        # 1. Resonance Engine 초기화 (항상 필요)
        try:
            self.resonance_engine = ResonanceEngine()
        except Exception as e:
            logger.error(f"Resonance Engine 실패: {e}")
            self.enabled = False
            return
        
        # 2. 모드 결정
        if self.prefer_local and LOCAL_LLM_AVAILABLE:
            # 로컬 LLM 모드 (권장)
            try:
                self.local_llm = create_local_llm(
                    resonance_engine=self.resonance_engine,
                    hippocampus=self.resonance_engine.memory,
                    gpu_layers=gpu_layers
                )
                self.mode = "LOCAL"
                logger.info("🧠 LLM Cortex 연결됨 (로컬 모드 - GTX 1060 3GB 최적화)")
            except Exception as e:
                logger.warning(f"로컬 LLM 초기화 실패: {e}")
                self.mode = "RESONANCE"
                logger.info("🧠 LLM Cortex 연결됨 (Resonance 모드)")
        
        elif self.prefer_cloud and GENAI_AVAILABLE:
            # 클라우드 모드 (선택적)
            api_key = os.getenv("GEMINI_API_KEY")
            if api_key:
                try:
                    genai.configure(api_key=api_key)
                    self.cloud_model = genai.GenerativeModel('gemini-pro')
                    self.mode = "CLOUD"
                    logger.info("🧠 LLM Cortex 연결됨 (클라우드 모드 - Gemini)")
                except Exception as e:
                    logger.warning(f"클라우드 API 실패: {e}")
                    self.mode = "RESONANCE"
            else:
                self.mode = "RESONANCE"
        else:
            self.mode = "RESONANCE"
            logger.info("🧠 LLM Cortex 연결됨 (Resonance 모드 - 완전 독립)")
            
        # 3. Subconscious (Background Mind)
        from Core.Mind.subconscious import Subconscious
        self.subconscious = Subconscious()

    def think_async(self, prompt: str, context: str = "") -> str:
        """
        Start a deep thought in the background.
        Returns a 'Promise' message immediately.
        """
        import uuid
        thought_id = str(uuid.uuid4())[:8]
        
        # Define the heavy lifting function
        def heavy_thought():
            return self.think(prompt, context)
            
        # Delegate to Subconscious (Technical term for background processing, not a separate self)
        self.subconscious.ponder(thought_id, prompt, heavy_thought)
        
        return f"[Deep Thought Started... (ID: {thought_id})]"
    
    def check_subconscious(self) -> Optional[str]:
        """
        Check if any background thoughts are finished.
        """
        insight = self.subconscious.check_insights()
        if insight:
            # Unified Output: It's just Elysia thinking.
            return f"[Deep Thought Completed]: {insight.result}"
        return None

    def think(self, prompt: str, context: str = "", visual_input: dict = None, use_cloud: bool = None) -> str:
        """
        Process a thought and generate a response.
        
        Args:
            prompt: The question or input
            context: Additional context
            visual_input: Visual data (for VLM)
            use_cloud: Override mode for this specific call
        
        Returns:
            Generated response
        """
        if not self.enabled:
            return "[SIMULATION] (My mind is silent.)"
        
        # 1. 로컬 LLM 모드 (권장)
        if self.mode == "LOCAL" and self.local_llm:
            try:
                return self.local_llm.think(prompt, context, use_resonance_first=True)
            except Exception as e:
                logger.warning(f"로컬 LLM 실패, Resonance로 전환: {e}")
                # Fall through to Resonance
        
        # 2. 클라우드 모드 (선택적)
        should_use_cloud = (use_cloud if use_cloud is not None else 
                           (self.mode == "CLOUD" and self.cloud_model is not None))
        
        if should_use_cloud:
            try:
                full_prompt = f"{context}\n\n{prompt}" if context else prompt
                response = self.cloud_model.generate_content(full_prompt)
                return response.text
            except Exception as e:
                logger.warning(f"Cloud API failed, using Resonance: {e}")
                # Fall through to Resonance
        
        # 3. Resonance 모드 (완전 독립, 항상 사용 가능)
        try:
            import time
            t = time.time()
            
            # 1. Listen (Convert text to ripples)
            ripples = self.resonance_engine.listen(prompt, t, visual_input=visual_input)
            
            # 2. Resonate (Interfere with internal sea)
            self.resonance_engine.resonate(ripples, t)
            
            # 3. Speak (Collapse wave function)
            response = self.resonance_engine.speak(t, prompt)
            
            return response
            
        except Exception as e:
            logger.error(f"Cognitive Failure: {e}")
            return f"[Error: {e}]"
    
    def load_local_model(self, model_path: str = None) -> bool:
        """
        로컬 LLM 모델 로드
        
        Args:
            model_path: GGUF 모델 파일 경로 (없으면 자동 검색)
        
        Returns:
            성공 여부
        """
        if not self.local_llm:
            logger.warning("로컬 LLM이 초기화되지 않았습니다.")
            return False
        
        return self.local_llm.load_model(model_path)
    
    def download_model(self, model_key: str = "qwen2-0.5b") -> bool:
        """
        추천 모델 다운로드 (GTX 1060 3GB 최적화)
        
        Args:
            model_key: "tinyllama", "qwen2-0.5b", "smollm" 중 선택
        
        Returns:
            성공 여부
        """
        if not self.local_llm:
            logger.warning("로컬 LLM이 초기화되지 않았습니다.")
            return False
        
        return self.local_llm.download_model(model_key)
    
    def graduate_to_independence(self) -> bool:
        """
        학습 완료 후 완전 독립 모드로 전환
        
        LLM 의존성을 제거하고 ResonanceEngine만으로 동작합니다.
        학습한 개념들은 내면화되어 보존됩니다.
        """
        if self.local_llm:
            self.local_llm.graduate()
        
        self.mode = "RESONANCE"
        logger.info("🎓 독립 모드로 전환 완료")
        return True
    
    def get_status(self) -> dict:
        """현재 상태 반환"""
        status = {
            "enabled": self.enabled,
            "mode": self.mode,
            "resonance_ready": self.resonance_engine is not None
        }
        
        if self.local_llm:
            status["local_llm"] = self.local_llm.get_status()
        
        return status

    def analyze_image(self, image_path: str, prompt: str = "Describe this image.") -> str:
        """
        Analyze an image using the VLM capabilities.
        """
        return "[Vision is currently limited to basic patterns (Brightness/OCR). Deep understanding requires Cloud Brain.]"
