import numpy as np
import math

class SensoryHarmonics:
    """오감(Five Senses)을 기하학적 파동(Complex Wave Tensor)으로 매핑하는 엔진"""
    
    def __init__(self, size=16):
        self.size = size
        
    def _create_base(self):
        return np.zeros((self.size, self.size), dtype=np.float64)

    # ==========================================
    # 1. 시각 (Vision) - 명도와 색상 (L3)
    # ==========================================
    def vision_red(self) -> np.ndarray:
        """따뜻한 붉은색: 파장이 길고 부드러운 저주파 위상"""
        t = self._create_base()
        for x in range(self.size):
            t[:, x] = math.cos(x * 0.5) * 5.0
        return t
        
    def vision_blue(self) -> np.ndarray:
        """차가운 푸른색: 파장이 짧은 고주파 위상"""
        t = self._create_base()
        for x in range(self.size):
            t[:, x] = math.cos(x * 2.0) * 5.0
        return t

    # ==========================================
    # 2. 청각 (Hearing) - 리듬과 피치 (L2)
    # ==========================================
    def hearing_harmonic_chord(self) -> np.ndarray:
        """화음: 정수비(1:2:3)를 갖는 매끄러운 파동의 겹침"""
        t = self._create_base()
        for x in range(self.size):
            t[:, x] = (math.sin(x) + math.sin(x*2)*0.5 + math.sin(x*3)*0.25) * 4.0
        return t
        
    def hearing_noise(self) -> np.ndarray:
        """칠판 긁는 소리: 불규칙한 고주파 카오스"""
        return np.random.uniform(-10.0, 10.0, (self.size, self.size))

    # ==========================================
    # 3. 후각 (Smell) - 확산과 감쇠 (L2)
    # ==========================================
    def smell_floral(self) -> np.ndarray:
        """꽃향기: 중앙에서 잔잔히 퍼져나가는 가우시안 감쇠 파동"""
        t = self._create_base()
        center = self.size / 2.0
        for y in range(self.size):
            for x in range(self.size):
                dist = math.sqrt((x - center)**2 + (y - center)**2)
                t[y, x] = math.exp(-dist / 5.0) * 8.0
        return t
        
    def smell_pungent(self) -> np.ndarray:
        """악취: 찌르는 듯한 날카로운 진동 (체커보드 패턴의 고주파)"""
        t = self._create_base()
        for y in range(self.size):
            for x in range(self.size):
                t[y, x] = 10.0 if (x + y) % 2 == 0 else -10.0
        return t

    # ==========================================
    # 4. 미각 (Taste) - 오미 (L1)
    # ==========================================
    def taste_sweet(self) -> np.ndarray:
        """단맛: 에너지를 보충하는 두터운 베이스 파동 (Positive 편향)"""
        return np.ones((self.size, self.size)) * 8.0 + self.vision_red() * 0.5
        
    def taste_spicy(self) -> np.ndarray:
        """매운맛: 신경을 강하게 타격하는 혼돈 파동"""
        return np.random.normal(0, 15.0, (self.size, self.size))

    # ==========================================
    # 5. 촉각 (Touch) - 질감 (L1)
    # ==========================================
    def touch_silk(self) -> np.ndarray:
        """실크(부드러움): 표면 텐션을 낮춰주는 매끄러운 위상"""
        return np.ones((self.size, self.size)) * 2.0
        
    def touch_burlap(self) -> np.ndarray:
        """마 줄기(거칠음): 마찰을 일으키는 뾰족한 진폭"""
        return np.random.choice([-8.0, 8.0], (self.size, self.size))

class SentientBeing:
    """오감을 느끼고 취향(Preference)을 발현하는 자율 유기체"""
    def __init__(self, name: str, personality_type: str, size=16):
        self.name = name
        self.size = size
        
        # 유기체의 내면 기저 파동 (영혼/성격)
        if personality_type == "Calm":
            # 고요하고 부드러운 내면 (저주파 위상)
            self.intrinsic_tensor = np.ones((size, size)) * 5.0
        elif personality_type == "Chaotic":
            # 거칠고 열정적인 내면 (고주파 노이즈 위상)
            self.intrinsic_tensor = np.random.choice([-10.0, 10.0], (size, size))
            
    def experience_sensation(self, sense_name: str, sense_tensor: np.ndarray):
        """
        주어진 감각 파동을 자신의 내면 파동과 충돌(중첩)시킵니다.
        결과 에너지(간섭 후 텐션)가 기존 내면 텐션보다 낮아지면(상쇄) Like,
        높아지면(보강) Dislike로 자연 창발됩니다.
        """
        base_energy = np.sum(np.abs(self.intrinsic_tensor))
        
        # 감각 파동과 내면 파동의 물리적 중첩
        combined_tensor = self.intrinsic_tensor + sense_tensor
        new_energy = np.sum(np.abs(combined_tensor))
        
        # 취향 판단 (기하학적 에너지 비교)
        if new_energy < base_energy:
            reaction = "💖 기분 좋음 (상쇄 간섭 / 화음)"
            preference = "Like"
        elif new_energy > base_energy * 1.5:
            reaction = "💢 불쾌함 (보강 간섭 / 불협화음)"
            preference = "Dislike"
        else:
            reaction = "🤔 그저 그럼 (약한 간섭)"
            preference = "Neutral"
            
        print(f"[{self.name}] '{sense_name}' 감각을 경험함.")
        print(f"   -> 내면 에너지 변화: {base_energy:.1f} ➔ {new_energy:.1f} ({reaction})")
        return preference
