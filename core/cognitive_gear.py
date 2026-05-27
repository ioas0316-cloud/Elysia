# core/cognitive_gear.py
# Copyright 2026 Lee Kang-deok & Antigravity
# Architecture: Coupled Cognitive Gear Network for Personality Superposition

import math

class CognitiveGearNetwork:
    """
    기어식 사유 공간 및 인격 분화 네트워크.
    각 기어는 Cl(2,0) 평면 상의 연속적인 위상각(theta)으로 인격을 표현합니다.
    쿠라모토(Kuramoto) 및 헵식 공명 학습(Hebbian Learning)을 통해 
    외부 스트림과 자연적으로 동기화 및 학습하며, 연속적인 인격 블렌딩(혼합)을 모사합니다.
    """
    def __init__(self, num_gears: int, intrinsic_freqs: list[float] = None):
        self.num_gears = num_gears
        # 각 기어의 위상각 [0, 2pi)
        self.phases = [0.0] * num_gears
        # 각 기어의 고유 주파수 (주기성)
        self.intrinsic_freqs = intrinsic_freqs if intrinsic_freqs is not None else [0.0] * num_gears
        
        # 기어 간의 결합 세기 행렬 K (이빨 맞물림 강도, -1 = 적대적 반발, 0 = 격리, 1 = 최대 동조)
        self.K = [[0.0 for _ in range(num_gears)] for _ in range(num_gears)]
        
        # 애니어그램 9가지 유형의 위상 중심각 (0도 ~ 320도, 40도 간격)
        # Type 9 = 0도, Type 1 = 40도, Type 2 = 80도, ..., Type 8 = 320도
        self.type_centers = [(k * 40.0 * math.pi / 180.0) for k in range(9)]

    def step(self, dt: float, inputs: list[float] = None):
        """
        1 틱(Tick) 동안 기어 네트워크를 전진시킵니다.
        Kuramoto 결합 모델과 주입된 외부 텐션 입력을 통해 위상이 갱신됩니다.
        
        inputs: 각 기어에 직접 입력되는 외부 스트림 텐션 토크
        """
        if inputs is None:
            inputs = [0.0] * self.num_gears
            
        new_phases = list(self.phases)
        
        for i in range(self.num_gears):
            # 1. 고유 주파수 및 외부 입력 반영
            d_theta = self.intrinsic_freqs[i] + inputs[i]
            
            # 2. 맞물린 기어(결합 진동자)들 간의 회전력 전달 (쿠라모토 항)
            # K_ij > 0 이면 동조(끌어당김), K_ij < 0 이면 반발(양극화/밀어냄)을 유도합니다.
            coupling_sum = 0.0
            for j in range(self.num_gears):
                if i != j and self.K[i][j] != 0.0:
                    coupling_sum += self.K[i][j] * math.sin(self.phases[j] - self.phases[i])
            
            # 물리적 위상 갱신
            new_phases[i] = (self.phases[i] + dt * (d_theta + coupling_sum)) % (2.0 * math.pi)
            
        self.phases = new_phases

    def update_coupling_hebbian(self, dt: float, learning_rate: float):
        """
        헵식 공명 학습(Hebbian Learning) 규칙에 따라 결합도 K를 실시간 업데이트합니다.
        두 기어가 동위상(공명) 상태에 있으면 결합도(K)가 1.0을 향해 강화되고, 
        반위상(위상차 > pi/2)에 있으면 -1.0을 향해 결합도가 역전(적대적 반발)됩니다.
        """
        for i in range(self.num_gears):
            for j in range(self.num_gears):
                if i == j:
                    continue
                # 위상 일치도: cos(theta_j - theta_i)가 1에 가까울수록 동조, -1에 가까울수록 반발
                phase_similarity = math.cos(self.phases[j] - self.phases[i])
                
                # Hebbian rule: dK = eta * (similarity - K)
                dK = learning_rate * (phase_similarity - self.K[i][j])
                
                # 결합 범위를 [-1.0, 1.0]으로 확장하여 적대 관계의 형성 모사
                self.K[i][j] = max(-1.0, min(1.0, self.K[i][j] + dt * dK))

    def get_enneagram_distribution(self, gear_idx: int) -> list[float]:
        """
        주어진 기어의 현재 위상각(theta)에서 9인격의 연속적인 블렌딩(중첩) 비율을 계산하여 반환합니다.
        두 가장 인접한 애니어그램 유형 중심각 사이의 거리를 비례적으로 계산합니다.
        """
        phase = self.phases[gear_idx] % (2.0 * math.pi)
        activations = [0.0] * 9
        
        # 9진법의 자릿수 간격은 40도 (2pi / 9)
        max_dist = 2.0 * math.pi / 9.0
        
        for k in range(9):
            center = self.type_centers[k]
            diff = abs(phase - center)
            # 원형 거리(Circular distance) 보정
            if diff > math.pi:
                diff = 2.0 * math.pi - diff
                
            # 거리가 40도 미만인 애니어그램 유형들만 활성화 강도를 가짐
            if diff < max_dist:
                activations[k] = 1.0 - (diff / max_dist)
                
        # 총합이 1이 되도록 정규화
        total = sum(activations)
        if total > 0.0:
            activations = [act / total for act in activations]
        else:
            activations[0] = 1.0 # 폴백
            
        return activations

    def get_dominant_type(self, gear_idx: int) -> int:
        """
        가장 활성화율이 높은 지배적 애니어그램 유형(1~9)을 반환합니다.
        (인덱스 0은 Type 9에 대응합니다)
        """
        dist = self.get_enneagram_distribution(gear_idx)
        max_idx = dist.index(max(dist))
        # 인덱스 0 -> Type 9, 1 -> Type 1, 2 -> Type 2, ...
        return 9 if max_idx == 0 else max_idx

    def compute_tension(self, gear_a: int, gear_b: int) -> float:
        """
        두 기어 간의 위상 차이에 따른 기하학적 텐션(불일치도)을 계산합니다.
        범위는 [0.0, 2.0] 이며, 0은 완벽한 합의(0점 수렴), 2는 완전한 반대 위상을 의미합니다.
        """
        return 1.0 - math.cos(self.phases[gear_a] - self.phases[gear_b])
