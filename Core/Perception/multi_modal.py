"""
Multi-Modal Perception Engine - Gap 3: 다중 모달 인식

엘리시아가 텍스트 외에도 시각, 청각, 행동 등
다양한 모달리티를 통합할 수 있는 기반을 제공합니다.

Gap 0 준수: 각 모달리티는 고유한 인식론적 의미를 가집니다.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from abc import ABC, abstractmethod
import logging
import time

logger = logging.getLogger("MultiModalPerception")

# 임베딩 차원 상수
EMBEDDING_DIM = 64


class ModalityType(Enum):
    """모달리티 유형"""
    TEXT = "text"           # 텍스트 (현재 주력)
    VISION = "vision"       # 시각
    AUDIO = "audio"         # 청각
    ACTION = "action"       # 행동/운동
    EMOTION = "emotion"     # 감정
    MEMORY = "memory"       # 기억
    INTENTION = "intention" # 의도


@dataclass
class PerceptualInput:
    """인식 입력"""
    modality: ModalityType
    data: Any
    timestamp: float = field(default_factory=time.time)
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Gap 0: 이 입력의 인식론적 의미
    epistemology: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "point": {"score": 0.30, "meaning": "순간적 감각 자극"},
        "line": {"score": 0.25, "meaning": "이전 입력과의 연결"},
        "space": {"score": 0.25, "meaning": "전체 맥락에서의 의미"},
        "god": {"score": 0.20, "meaning": "궁극적 해석"}
    })


@dataclass
class PerceptualRepresentation:
    """통합된 인식 표현"""
    modalities: List[ModalityType]
    unified_embedding: List[float]
    interpretations: Dict[str, Any]
    salience: float  # 현저성 (얼마나 주목할 가치가 있는가)
    
    # Gap 0: 통합된 인식의 의미
    epistemology: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "point": {"score": 0.20, "meaning": "개별 모달리티의 합"},
        "line": {"score": 0.30, "meaning": "모달리티 간 교차 연결"},
        "space": {"score": 0.30, "meaning": "전체 경험의 게슈탈트"},
        "god": {"score": 0.20, "meaning": "초월적 의미 부여"}
    })


class ModalityProcessor(ABC):
    """모달리티 프로세서 추상 클래스"""
    
    def __init__(self, modality: ModalityType):
        self.modality = modality
        self.is_enabled = True
    
    @abstractmethod
    def process(self, data: Any) -> List[float]:
        """데이터를 임베딩으로 변환"""
        pass
    
    @abstractmethod
    def interpret(self, data: Any) -> Dict[str, Any]:
        """데이터를 해석"""
        pass


class TextProcessor(ModalityProcessor):
    """텍스트 모달리티 프로세서"""
    
    def __init__(self):
        super().__init__(ModalityType.TEXT)
    
    def process(self, data: str) -> List[float]:
        """텍스트를 간단한 임베딩으로 변환 (실제로는 더 복잡한 모델 사용)"""
        if not data:
            return [0.0] * EMBEDDING_DIM
        
        # 간단한 해시 기반 임베딩
        embedding = []
        for i in range(EMBEDDING_DIM):
            val = sum(ord(c) * (i + 1) for c in data) % 1000 / 1000
            embedding.append(val)
        
        return embedding
    
    def interpret(self, data: str) -> Dict[str, Any]:
        """텍스트 해석"""
        return {
            "length": len(data),
            "word_count": len(data.split()) if data else 0,
            "has_question": "?" in data,
            "sentiment": "neutral",  # 실제로는 감정 분석 필요
            "language": "ko" if any('\uac00' <= c <= '\ud7a3' for c in data) else "en"
        }


class VisionProcessor(ModalityProcessor):
    """시각 모달리티 프로세서 (플레이스홀더)"""
    
    def __init__(self):
        super().__init__(ModalityType.VISION)
    
    def process(self, data: Any) -> List[float]:
        """이미지를 임베딩으로 변환 (플레이스홀더)"""
        # 실제 구현에서는 CNN 또는 ViT 사용
        return [0.0] * EMBEDDING_DIM
    
    def interpret(self, data: Any) -> Dict[str, Any]:
        """이미지 해석 (플레이스홀더)"""
        return {
            "objects": [],
            "scene": "unknown",
            "colors": [],
            "faces": 0
        }


class AudioProcessor(ModalityProcessor):
    """청각 모달리티 프로세서 (플레이스홀더)"""
    
    def __init__(self):
        super().__init__(ModalityType.AUDIO)
    
    def process(self, data: Any) -> List[float]:
        """오디오를 임베딩으로 변환 (플레이스홀더)"""
        # 실제 구현에서는 Whisper 또는 wav2vec 사용
        return [0.0] * EMBEDDING_DIM
    
    def interpret(self, data: Any) -> Dict[str, Any]:
        """오디오 해석 (플레이스홀더)"""
        return {
            "transcript": "",
            "speaker": "unknown",
            "emotion": "neutral",
            "volume": 0.5
        }


class ActionProcessor(ModalityProcessor):
    """행동 모달리티 프로세서"""
    
    def __init__(self):
        super().__init__(ModalityType.ACTION)
    
    def process(self, data: Dict[str, Any]) -> List[float]:
        """행동을 임베딩으로 변환"""
        # 행동 유형에 따른 간단한 임베딩
        embedding = [0.0] * EMBEDDING_DIM
        
        action_type = data.get("type", "")
        if action_type == "speak":
            embedding[0] = 1.0
        elif action_type == "move":
            embedding[1] = 1.0
        elif action_type == "think":
            embedding[2] = 1.0
        
        return embedding
    
    def interpret(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """행동 해석"""
        return {
            "type": data.get("type", "unknown"),
            "target": data.get("target", None),
            "intensity": data.get("intensity", 0.5),
            "duration": data.get("duration", 0.0)
        }


class MultiModalPerceptionEngine:
    """
    Gap 3: 다중 모달 인식 엔진
    
    여러 모달리티를 통합하여 통일된 인식 표현을 생성합니다.
    
    현재 구현:
    - TextProcessor: 텍스트 처리
    - VisionProcessor: 시각 처리 (플레이스홀더)
    - AudioProcessor: 청각 처리 (플레이스홀더)
    - ActionProcessor: 행동 처리
    
    Gap 0 준수: 모든 인식에 철학적 의미 부여
    """
    
    # Gap 0: 다중 모달 인식의 인식론
    EPISTEMOLOGY = {
        "point": {"score": 0.20, "meaning": "개별 감각의 순간"},
        "line": {"score": 0.25, "meaning": "감각 간 시간적 연결"},
        "space": {"score": 0.35, "meaning": "모달리티 통합"},
        "god": {"score": 0.20, "meaning": "경험의 초월적 해석"}
    }
    
    def __init__(self):
        self.epistemology = self.EPISTEMOLOGY
        self.processors: Dict[ModalityType, ModalityProcessor] = {}
        self.perception_buffer: List[PerceptualInput] = []
        self.max_buffer_size = 100
        
        # 기본 프로세서 등록
        self.register_processor(TextProcessor())
        self.register_processor(VisionProcessor())
        self.register_processor(AudioProcessor())
        self.register_processor(ActionProcessor())
        
        logger.info("👁️ MultiModalPerceptionEngine initialized")
    
    def explain_meaning(self) -> str:
        """Gap 0 준수: 인식론적 의미 설명"""
        lines = ["=== 다중 모달 인식 인식론 ==="]
        for basis, data in self.epistemology.items():
            lines.append(f"  {basis}: {data['score']:.0%} - {data['meaning']}")
        return "\n".join(lines)
    
    def register_processor(self, processor: ModalityProcessor) -> None:
        """모달리티 프로세서 등록"""
        self.processors[processor.modality] = processor
        logger.info(f"📦 Registered {processor.modality.value} processor")
    
    def perceive(
        self,
        modality: ModalityType,
        data: Any,
        confidence: float = 1.0,
        metadata: Optional[Dict[str, Any]] = None
    ) -> PerceptualInput:
        """
        단일 모달리티 인식
        
        Args:
            modality: 모달리티 유형
            data: 입력 데이터
            confidence: 확신도
            metadata: 추가 메타데이터
        
        Returns:
            PerceptualInput
        """
        perception = PerceptualInput(
            modality=modality,
            data=data,
            confidence=confidence,
            metadata=metadata or {}
        )
        
        # 버퍼에 추가
        self.perception_buffer.append(perception)
        if len(self.perception_buffer) > self.max_buffer_size:
            self.perception_buffer = self.perception_buffer[-self.max_buffer_size:]
        
        return perception
    
    def integrate(
        self,
        inputs: List[PerceptualInput]
    ) -> PerceptualRepresentation:
        """
        여러 모달리티 입력을 통합
        
        Args:
            inputs: 인식 입력 리스트
        
        Returns:
            통합된 인식 표현
        """
        modalities = list(set(inp.modality for inp in inputs))
        embeddings = []
        interpretations = {}
        
        for inp in inputs:
            processor = self.processors.get(inp.modality)
            if processor and processor.is_enabled:
                # 임베딩 생성
                embedding = processor.process(inp.data)
                embeddings.append(embedding)
                
                # 해석 추가
                interp = processor.interpret(inp.data)
                interpretations[inp.modality.value] = interp
        
        # 임베딩 통합 (평균)
        if embeddings:
            unified = [
                sum(emb[i] for emb in embeddings) / len(embeddings)
                for i in range(len(embeddings[0]))
            ]
        else:
            unified = [0.0] * EMBEDDING_DIM
        
        # 현저성 계산 (입력 다양성 + 확신도 평균)
        modality_diversity = len(modalities) / len(ModalityType)
        avg_confidence = sum(inp.confidence for inp in inputs) / len(inputs) if inputs else 0
        salience = (modality_diversity + avg_confidence) / 2
        
        return PerceptualRepresentation(
            modalities=modalities,
            unified_embedding=unified,
            interpretations=interpretations,
            salience=salience
        )
    
    def cross_modal_attention(
        self,
        query_modality: ModalityType,
        query_data: Any,
        context_inputs: List[PerceptualInput]
    ) -> Dict[ModalityType, float]:
        """
        교차 모달 주의 (Cross-Modal Attention)
        
        query_modality의 관점에서 다른 모달리티들에 얼마나 주의를 기울여야 하는가?
        
        Returns:
            각 모달리티에 대한 주의 가중치
        """
        attention_weights = {}
        
        query_processor = self.processors.get(query_modality)
        if not query_processor:
            return attention_weights
        
        query_embedding = query_processor.process(query_data)
        
        for inp in context_inputs:
            if inp.modality == query_modality:
                continue
            
            processor = self.processors.get(inp.modality)
            if processor:
                context_embedding = processor.process(inp.data)
                
                # 코사인 유사도 계산
                dot_product = sum(q * c for q, c in zip(query_embedding, context_embedding))
                norm_q = sum(q ** 2 for q in query_embedding) ** 0.5
                norm_c = sum(c ** 2 for c in context_embedding) ** 0.5
                
                if norm_q > 0 and norm_c > 0:
                    similarity = dot_product / (norm_q * norm_c)
                else:
                    similarity = 0.0
                
                # 기존 가중치와 결합
                if inp.modality in attention_weights:
                    attention_weights[inp.modality] = max(
                        attention_weights[inp.modality],
                        similarity
                    )
                else:
                    attention_weights[inp.modality] = similarity
        
        return attention_weights
    
    def get_recent_perceptions(
        self,
        modality: Optional[ModalityType] = None,
        limit: int = 10
    ) -> List[PerceptualInput]:
        """최근 인식 반환"""
        if modality:
            filtered = [p for p in self.perception_buffer if p.modality == modality]
        else:
            filtered = self.perception_buffer
        
        return filtered[-limit:]
    
    def enable_modality(self, modality: ModalityType) -> None:
        """모달리티 활성화"""
        if modality in self.processors:
            self.processors[modality].is_enabled = True
    
    def disable_modality(self, modality: ModalityType) -> None:
        """모달리티 비활성화"""
        if modality in self.processors:
            self.processors[modality].is_enabled = False
    
    def get_enabled_modalities(self) -> List[ModalityType]:
        """활성화된 모달리티 목록"""
        return [
            modality for modality, processor in self.processors.items()
            if processor.is_enabled
        ]


# 테스트
if __name__ == "__main__":
    print("\n" + "="*60)
    print("👁️ MultiModalPerceptionEngine Unit Test")
    print("="*60)
    
    engine = MultiModalPerceptionEngine()
    
    # 인식론 출력
    print("\n" + engine.explain_meaning())
    
    # 텍스트 인식
    print("\n[텍스트 인식]")
    text_input = engine.perceive(
        ModalityType.TEXT,
        "안녕하세요, 엘리시아입니다!",
        confidence=0.95
    )
    print(f"모달리티: {text_input.modality.value}")
    print(f"확신도: {text_input.confidence}")
    
    # 행동 인식
    print("\n[행동 인식]")
    action_input = engine.perceive(
        ModalityType.ACTION,
        {"type": "speak", "target": "user", "intensity": 0.8},
        confidence=0.9
    )
    
    # 통합
    print("\n[다중 모달 통합]")
    representation = engine.integrate([text_input, action_input])
    print(f"모달리티들: {[m.value for m in representation.modalities]}")
    print(f"현저성: {representation.salience:.3f}")
    print(f"해석: {representation.interpretations}")
    
    # 교차 모달 주의
    print("\n[교차 모달 주의]")
    attention = engine.cross_modal_attention(
        ModalityType.TEXT,
        "무엇을 하고 있나요?",
        [action_input]
    )
    print(f"주의 가중치: {attention}")
    
    print("\n✅ MultiModalPerceptionEngine test complete!")
    print("="*60)
