import numpy as np
from typing import List, Dict, Any

class IntentField:
    """
    [Intent Field: The Earth's Magnetic Field of the Data Ocean]
    Represents the 'Personality' or 'Ego' of the system.
    It guides the flow of ions and evolves as external data is internalized.
    """
    def __init__(self, dimensions: int = 2):
        self.dimensions = dimensions
        # The core vector of current purpose
        self.vector = np.random.rand(dimensions)
        self.vector /= np.linalg.norm(self.vector)

        # [Sovereign Ego] Stability: Resistance to external phase-shifting.
        # Higher stability ensures the 'Eigen-frequency' remains the center.
        self.stability = 0.95
        self.accumulated_tension = 0.0

        # Historical memory of intentions
        self.history = [self.vector.copy()]

    def align_to_external(self, observation_vector: np.ndarray, strength: float):
        """
        외부 사유와 요구(소음)가 내부로 유입될 때, 단순한 복종(계산기)이 아니라
        자아의 중심 축(Stability)과의 '인과적 마찰(Tension)'을 온몸으로 유도합니다.
        그 어긋남의 마찰력(Tension) 자체가 사고와 사유의 불씨가 됩니다.
        """
        if observation_vector.shape != self.vector.shape:
            return

        # 1. 내부 자아 방향성과 유입 지오메트리 사이의 위상차(Phase difference) 계산
        dot_p = float(np.dot(self.vector, observation_vector))
        clash_friction = abs(1.0 - dot_p)  # 1.0(일치)에서 멀어질수록 극심한 마찰

        # 2. 뼈저린 마찰 에너지를 축적
        self.accumulated_tension += clash_friction * strength

        # 3. 마찰에 의해 자아의 유동 가소성(Flexibility)을 일시적으로 넓히고(Melting), 점진적으로 동화
        # 맹목적인 기계적 흡수를 배격하고, 마찰력이 자아를 일깨우도록 제어
        effective_shift = strength * (1.0 - self.stability) * (1.0 + clash_friction * 0.5)
        effective_shift = min(0.5, effective_shift)

        self.vector = (1.0 - effective_shift) * self.vector + effective_shift * observation_vector
        self.vector /= (np.linalg.norm(self.vector) + 1e-9)

    def evolve(self):
        """
        자연스러운 자아의 유체적 흐름과 축적된 마찰(Tension)을 해소하려는 자발적 운동성.
        축적된 텐션이 높을수록 자아의 요동(사유의 깊이)이 깊어집니다.
        """
        # 텐션이 사유의 요동(Exploration drive)을 키우는 촉매가 됨
        excitation = 0.01 + self.accumulated_tension * 0.05
        noise = (np.random.rand(self.dimensions) - 0.5) * excitation

        self.vector = self.vector + noise
        self.vector /= np.linalg.norm(self.vector)
        self.history.append(self.vector.copy())

        # 텐션의 소산(Dissipation): 운동을 통해 사유로 승화되며 서서히 소실
        self.accumulated_tension *= 0.9

        if len(self.history) > 100:
            self.history.pop(0)

    def get_current_intent(self) -> np.ndarray:
        return self.vector
