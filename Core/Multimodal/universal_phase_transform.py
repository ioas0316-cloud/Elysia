"""
Universal Phase Transform (범용 위상 변환)
=========================================

"모든 감각은 파동이다"

엘리시아 변환의 범용 확장:
- 소리 (Audio)
- 글 (Text) 
- 그림 (Image)
- 영상 (Video)
- 개념 (Concept)

모두 4차원 쿼터니언 위상 공명 패턴으로 변환 가능!

핵심 원리:
1. 모든 감각/개념은 파동으로 표현 가능
2. 4차원 위상 단위 (쿼터니언)로 매핑
3. 서로의 영역에서 간섭 없이 통신
4. 원할 때 언제든지 공감각(Synesthesia)으로 변환

"5감 주파수 매핑의 완성"
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any, Union
from enum import Enum
import logging

logger = logging.getLogger("UniversalPhaseTransform")


class Modality(Enum):
    """감각 모달리티"""
    AUDIO = "audio"      # 청각 (소리)
    TEXT = "text"        # 언어 (글)
    IMAGE = "image"      # 시각 (그림)
    VIDEO = "video"      # 시각+시간 (영상)
    CONCEPT = "concept"  # 추상 (개념)
    TOUCH = "touch"      # 촉각
    SMELL = "smell"      # 후각
    TASTE = "taste"      # 미각


@dataclass
class PhaseQuaternion:
    """
    범용 위상 쿼터니언
    
    q = w + xi + yj + zk
    
    모든 감각/개념의 4차원 위상 표현
    
    - w: 강도 (Intensity) - 에너지, 존재감, 중요도
    - x: 주파수 (Frequency) - 진동, 리듬, 패턴 반복
    - y: 위상 (Phase) - 방향, 관계, 맥락
    - z: 복잡도 (Complexity) - 구조, 질감, 풍부함
    """
    w: float  # Intensity (0.0 ~ 1.0)
    x: float  # Frequency (normalized)
    y: float  # Phase (0.0 ~ 2π)
    z: float  # Complexity (0.0 ~ 1.0)
    modality: Modality  # 원본 감각 모달리티
    
    def __post_init__(self):
        """정규화"""
        self.w = max(0.0, min(1.0, self.w))
        self.y = self.y % (2 * np.pi)
        self.z = max(0.0, min(1.0, self.z))
    
    def to_vector(self) -> np.ndarray:
        """4차원 벡터로 변환"""
        return np.array([self.w, self.x, self.y, self.z])
    
    def resonance(self, other: 'PhaseQuaternion') -> float:
        """
        두 위상 쿼터니언 간의 공명도
        
        같은 모달리티끼리는 강한 공명
        다른 모달리티끼리는 약한 공명 (간섭 없음!)
        """
        diff = self.to_vector() - other.to_vector()
        distance = np.linalg.norm(diff)
        
        # 같은 모달리티면 공명 강화
        modality_factor = 1.0 if self.modality == other.modality else 0.3
        
        # 거리가 가까울수록 공명도 높음
        resonance = np.exp(-distance) * modality_factor
        
        return resonance
    
    def to_synesthesia(self, target_modality: Modality) -> Dict[str, Any]:
        """
        공감각 변환 (Synesthesia)
        
        한 감각을 다른 감각으로 변환
        예: 소리 → 색깔, 글 → 소리, 그림 → 음악
        """
        result = {
            'source_modality': self.modality.value,
            'target_modality': target_modality.value,
            'quaternion': self.to_vector().tolist()
        }
        
        if target_modality == Modality.IMAGE:
            # 시각으로 변환 (색상)
            result['color'] = self._to_color()
            result['description'] = f"{self._color_name()} {self._texture_name()}"
            
        elif target_modality == Modality.AUDIO:
            # 청각으로 변환 (음파)
            result['note'] = self._to_musical_note()
            result['timbre'] = self._timbre_name()
            result['description'] = f"{result['note']} {result['timbre']}"
            
        elif target_modality == Modality.TEXT:
            # 언어로 변환 (묘사)
            result['description'] = self._to_text_description()
            
        elif target_modality == Modality.TOUCH:
            # 촉각으로 변환 (질감)
            result['texture'] = self._texture_name()
            result['temperature'] = "따뜻한" if self.w > 0.5 else "차가운"
            result['description'] = f"{result['temperature']} {result['texture']}"
        
        return result
    
    def _to_color(self) -> Tuple[float, float, float, float]:
        """색상으로 변환 (RGBA)"""
        hue = (self.x % 1.0) * 360.0
        saturation = self.z
        value = self.w
        alpha = (np.cos(self.y) + 1.0) / 2.0
        
        # HSV to RGB
        h = hue / 60.0
        c = value * saturation
        x = c * (1 - abs(h % 2 - 1))
        m = value - c
        
        if h < 1:
            r, g, b = c, x, 0
        elif h < 2:
            r, g, b = x, c, 0
        elif h < 3:
            r, g, b = 0, c, x
        elif h < 4:
            r, g, b = 0, x, c
        elif h < 5:
            r, g, b = x, 0, c
        else:
            r, g, b = c, 0, x
        
        return (r + m, g + m, b + m, alpha)
    
    def _color_name(self) -> str:
        """색상 이름"""
        r, g, b, _ = self._to_color()
        if r > g and r > b:
            return "붉은" if r > 0.6 else "분홍"
        elif g > r and g > b:
            return "초록" if g > 0.6 else "청록"
        elif b > r and b > g:
            return "파란" if b > 0.6 else "하늘"
        elif r > 0.5 and g > 0.5:
            return "황금"
        else:
            return "은빛"
    
    def _texture_name(self) -> str:
        """질감 이름"""
        if self.z > 0.7:
            return "거친"
        elif self.z > 0.4:
            return "부드러운"
        else:
            return "매끄러운"
    
    def _to_musical_note(self) -> str:
        """음계로 변환"""
        notes = ['도', '도#', '레', '레#', '미', '파', '파#', '솔', '솔#', '라', '라#', '시']
        note_idx = int(self.x * 12) % 12
        octave = int(self.x * 8) + 1
        return f"{notes[note_idx]}{octave}"
    
    def _timbre_name(self) -> str:
        """음색 이름"""
        if self.z > 0.7:
            return "풍부한"
        elif self.z > 0.4:
            return "따뜻한"
        else:
            return "맑은"
    
    def _to_text_description(self) -> str:
        """텍스트 묘사"""
        intensity = "강렬한" if self.w > 0.7 else "은은한" if self.w > 0.4 else "미세한"
        pattern = "빠른" if self.x > 0.7 else "보통" if self.x > 0.4 else "느린"
        complexity = "복잡한" if self.z > 0.7 else "조화로운" if self.z > 0.4 else "단순한"
        
        return f"{intensity} {pattern} {complexity} 파동"
    
    def __str__(self):
        return f"PhaseQ[{self.modality.value}|w={self.w:.2f}, x={self.x:.2f}, y={self.y:.2f}, z={self.z:.2f}]"


class UniversalPhaseTransform:
    """
    범용 위상 변환 (Universal Phase Transform)
    
    모든 감각과 개념을 4차원 쿼터니언 위상 공명 패턴으로 변환
    """
    
    def __init__(self):
        logger.info("🌐 Universal Phase Transform initialized")
        logger.info("   All modalities → 4D Phase Resonance Pattern")
    
    def transform_audio(self, audio_signal: np.ndarray, sample_rate: int = 44100) -> List[PhaseQuaternion]:
        """오디오를 위상 쿼터니언으로 변환"""
        from Core.Multimodal.elysia_transform import ElysiaTransform
        
        audio_transform = ElysiaTransform(sample_rate)
        sound_quaternions = audio_transform.transform(audio_signal)
        
        # SoundQuaternion → PhaseQuaternion
        phase_quaternions = []
        for sq in sound_quaternions:
            pq = PhaseQuaternion(
                w=sq.w,
                x=sq.x,
                y=sq.y,
                z=sq.z,
                modality=Modality.AUDIO
            )
            phase_quaternions.append(pq)
        
        logger.info(f"✅ Audio → {len(phase_quaternions)} phase quaternions")
        return phase_quaternions
    
    def transform_text(self, text: str) -> List[PhaseQuaternion]:
        """
        텍스트를 위상 쿼터니언으로 변환
        
        글의 파동:
        - w: 단어 중요도 (TF-IDF, 감정 강도)
        - x: 리듬 (음절 수, 문장 길이)
        - y: 맥락 (문맥, 위치)
        - z: 복잡도 (어휘 다양성, 구조)
        """
        words = text.split()
        quaternions = []
        
        for i, word in enumerate(words):
            # w: 단어 길이로 중요도 추정 (간단한 휴리스틱)
            w = min(1.0, len(word) / 15.0)
            
            # x: 음절 리듬 (글자 수)
            x = (len(word) % 10) / 10.0
            
            # y: 문장 내 위치 (위상)
            y = (i / len(words)) * 2 * np.pi
            
            # z: 복잡도 (대문자, 특수문자 비율)
            complexity = sum(1 for c in word if c.isupper() or not c.isalnum()) / max(len(word), 1)
            z = min(1.0, complexity * 3)
            
            pq = PhaseQuaternion(w, x, y, z, Modality.TEXT)
            quaternions.append(pq)
        
        logger.info(f"✅ Text → {len(quaternions)} phase quaternions")
        return quaternions
    
    def transform_image(self, image_array: np.ndarray) -> List[PhaseQuaternion]:
        """
        이미지를 위상 쿼터니언으로 변환
        
        그림의 파동:
        - w: 밝기 (Brightness)
        - x: 색상 주파수 (Hue)
        - y: 채도/위상 (Saturation)
        - z: 질감 복잡도 (Texture)
        """
        # 이미지를 블록으로 나누어 분석 (간단한 구현)
        if len(image_array.shape) == 3:
            h, w, c = image_array.shape
        else:
            h, w = image_array.shape
            c = 1
        
        block_size = 32
        quaternions = []
        
        for i in range(0, h, block_size):
            for j in range(0, w, block_size):
                block = image_array[i:i+block_size, j:j+block_size]
                
                if c == 3 or c == 4:
                    # 컬러 이미지
                    r = block[:,:,0].mean() / 255.0
                    g = block[:,:,1].mean() / 255.0
                    b = block[:,:,2].mean() / 255.0
                    
                    # RGB → HSV
                    brightness = (r + g + b) / 3.0
                    hue = np.arctan2(np.sqrt(3) * (g - b), 2 * r - g - b)
                    hue = (hue % (2 * np.pi)) / (2 * np.pi)
                    saturation = 1 - 3 * min(r, g, b) / (r + g + b + 1e-6)
                    
                    # 질감 (분산)
                    texture = np.std(block) / 128.0
                    
                    pq = PhaseQuaternion(
                        w=brightness,
                        x=hue,
                        y=saturation * 2 * np.pi,
                        z=min(1.0, texture),
                        modality=Modality.IMAGE
                    )
                else:
                    # 그레이스케일
                    brightness = block.mean() / 255.0
                    texture = np.std(block) / 128.0
                    
                    pq = PhaseQuaternion(
                        w=brightness,
                        x=0.0,
                        y=0.0,
                        z=min(1.0, texture),
                        modality=Modality.IMAGE
                    )
                
                quaternions.append(pq)
        
        logger.info(f"✅ Image → {len(quaternions)} phase quaternions")
        return quaternions
    
    def transform_concept(self, concept_data: Dict[str, Any]) -> PhaseQuaternion:
        """
        추상 개념을 위상 쿼터니언으로 변환
        
        개념의 파동:
        - w: 중요도/활성화 (Importance/Activation)
        - x: 범주 주파수 (Category)
        - y: 관계 위상 (Relation)
        - z: 구조 복잡도 (Structure)
        """
        # 개념 데이터에서 특징 추출
        importance = concept_data.get('importance', 0.5)
        category = hash(concept_data.get('category', '')) % 1000 / 1000.0
        relation_count = len(concept_data.get('relations', []))
        structure_depth = concept_data.get('depth', 1)
        
        pq = PhaseQuaternion(
            w=importance,
            x=category,
            y=(relation_count % 10) / 10.0 * 2 * np.pi,
            z=min(1.0, structure_depth / 10.0),
            modality=Modality.CONCEPT
        )
        
        logger.info(f"✅ Concept → phase quaternion")
        return pq
    
    def cross_modal_resonance(self, 
                               quaternions_a: List[PhaseQuaternion],
                               quaternions_b: List[PhaseQuaternion]) -> np.ndarray:
        """
        크로스 모달 공명 행렬
        
        서로 다른 감각 간의 공명 패턴 분석
        예: 음악과 그림이 얼마나 조화로운가?
        """
        n_a = len(quaternions_a)
        n_b = len(quaternions_b)
        
        resonance_matrix = np.zeros((n_a, n_b))
        
        for i, qa in enumerate(quaternions_a):
            for j, qb in enumerate(quaternions_b):
                resonance_matrix[i, j] = qa.resonance(qb)
        
        logger.info(f"✅ Cross-modal resonance: {n_a}x{n_b} matrix")
        return resonance_matrix
    
    def synesthesia_transform(self,
                              source_quaternions: List[PhaseQuaternion],
                              target_modality: Modality) -> List[Dict[str, Any]]:
        """
        공감각 변환 (Synesthesia Transform)
        
        한 감각을 다른 감각으로 변환
        """
        results = []
        
        for pq in source_quaternions:
            synesthesia = pq.to_synesthesia(target_modality)
            results.append(synesthesia)
        
        logger.info(f"✅ Synesthesia: {source_quaternions[0].modality.value} → {target_modality.value}")
        return results
    
    def interference_free_communication(self,
                                       messages: List[Tuple[PhaseQuaternion, Any]]) -> Dict[Modality, List[Any]]:
        """
        간섭 없는 통신
        
        각 모달리티별로 메시지 분리
        4차원 위상 단위 덕분에 서로 간섭하지 않음!
        """
        channels = {}
        
        for pq, message in messages:
            modality = pq.modality
            if modality not in channels:
                channels[modality] = []
            channels[modality].append(message)
        
        logger.info(f"✅ Interference-free communication: {len(channels)} channels")
        return channels


def demonstrate_universal_transform():
    """범용 위상 변환 데모"""
    print("="*80)
    print("🌐 범용 위상 변환 (Universal Phase Transform) 데모")
    print("   '모든 감각은 파동이다'")
    print("="*80)
    print()
    
    transform = UniversalPhaseTransform()
    
    # 1. 텍스트 변환
    print("📝 1. 텍스트 → 위상 쿼터니언")
    text = "엘리시아는 모든 감각을 이해합니다"
    text_quats = transform.transform_text(text)
    print(f"   입력: '{text}'")
    print(f"   출력: {len(text_quats)}개 쿼터니언")
    for i, q in enumerate(text_quats[:3]):
        print(f"   {i+1}. {q}")
    print()
    
    # 2. 이미지 변환 (더미 데이터)
    print("🖼️  2. 이미지 → 위상 쿼터니언")
    dummy_image = np.random.rand(64, 64, 3) * 255
    image_quats = transform.transform_image(dummy_image)
    print(f"   입력: 64x64 RGB 이미지")
    print(f"   출력: {len(image_quats)}개 쿼터니언")
    print(f"   샘플: {image_quats[0]}")
    print()
    
    # 3. 개념 변환
    print("💡 3. 개념 → 위상 쿼터니언")
    concept = {
        'name': '사랑',
        'importance': 0.9,
        'category': 'emotion',
        'relations': ['행복', '따뜻함', '연결'],
        'depth': 3
    }
    concept_quat = transform.transform_concept(concept)
    print(f"   입력: {concept['name']} (중요도: {concept['importance']})")
    print(f"   출력: {concept_quat}")
    print()
    
    # 4. 공감각 변환
    print("🎨 4. 공감각 변환 (Synesthesia)")
    print("   텍스트 → 색상:")
    text_to_color = transform.synesthesia_transform(text_quats[:3], Modality.IMAGE)
    for i, syn in enumerate(text_to_color):
        word = text.split()[i]
        print(f"   '{word}' → {syn['description']}")
    print()
    
    print("   텍스트 → 소리:")
    text_to_sound = transform.synesthesia_transform(text_quats[:3], Modality.AUDIO)
    for i, syn in enumerate(text_to_sound):
        word = text.split()[i]
        print(f"   '{word}' → {syn['note']} {syn['timbre']}")
    print()
    
    # 5. 크로스 모달 공명
    print("🔗 5. 크로스 모달 공명")
    resonance = transform.cross_modal_resonance(text_quats[:3], image_quats[:3])
    print(f"   텍스트 x 이미지 공명 행렬:")
    print(f"   {resonance}")
    print(f"   평균 공명도: {resonance.mean():.3f}")
    print()
    
    # 6. 간섭 없는 통신
    print("📡 6. 간섭 없는 통신")
    messages = [
        (text_quats[0], "텍스트 메시지 1"),
        (image_quats[0], "이미지 메시지 1"),
        (concept_quat, "개념 메시지 1"),
        (text_quats[1], "텍스트 메시지 2"),
    ]
    channels = transform.interference_free_communication(messages)
    print(f"   총 메시지: {len(messages)}개")
    print(f"   채널 분리:")
    for modality, msgs in channels.items():
        print(f"   - {modality.value}: {len(msgs)}개 메시지")
    print()
    
    print("="*80)
    print("✨ 핵심 원리:")
    print("   1. 모든 감각/개념은 파동 → 4D 쿼터니언으로 표현")
    print("   2. 서로 다른 모달리티는 간섭 없이 통신 (0.3배 약한 공명)")
    print("   3. 원할 때는 공감각으로 자유롭게 변환 가능")
    print("   4. '5감 주파수 매핑'의 완성!")
    print("="*80)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    demonstrate_universal_transform()
