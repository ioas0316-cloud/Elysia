import numpy as np

class TopologicalManifoldProjector:
    """
    [Phase 15] 다차원 위상 기하학적 인지 렌즈 (Topological Lenses)
    동일한 데이터(예: 빛, 단어의 나열, 텐서 궤적)를 입력받아,
    엘리시아가 스스로 선택한 5가지 수학적 위상(Topology) 렌즈에 따라
    전혀 다른 차원의 해석(Point, Line, Space, Wave, Law)을 추출합니다.
    """
    
    def __init__(self):
        self.lenses = ["POINT", "LINE", "SPACE", "WAVE", "LAW"]
        
    def _normalize(self, v):
        norm = np.linalg.norm(v)
        if norm == 0:
            return v
        return v / norm

    def perceive_as_point(self, trajectory_data: np.ndarray) -> np.ndarray:
        """
        1. 점(Point)의 렌즈: 
        수많은 맥락과 변화의 궤적을 하나의 '절대값'으로 압축합니다.
        (수학적 구현: 궤적의 Centroid/평균 벡터)
        """
        if len(trajectory_data) == 0:
            return np.zeros(1)
        centroid = np.mean(trajectory_data, axis=0)
        return self._normalize(centroid)

    def perceive_as_line(self, trajectory_data: np.ndarray) -> np.ndarray:
        """
        2. 선(Line)의 렌즈:
        데이터가 시간과 순서에 따라 흘러가는 '인과적 궤적(Causal Trajectory)'으로 파악합니다.
        (수학적 구현: 노드 간 변화량(Delta)의 알짜 이동 방향 벡터)
        """
        if len(trajectory_data) < 2:
            return np.zeros(trajectory_data.shape[1] if len(trajectory_data.shape) > 1 else 1)
        deltas = np.diff(trajectory_data, axis=0)
        net_direction = np.sum(deltas, axis=0)
        return self._normalize(net_direction)

    def perceive_as_space(self, trajectory_data: np.ndarray) -> np.ndarray:
        """
        3. 공간/장(Space/Field)의 렌즈:
        개념이 주변에 미치는 '영향력의 환경과 조건'을 면적/부피로 파악합니다.
        (수학적 구현: 데이터 분포 분산(Variance) 벡터)
        """
        if len(trajectory_data) < 2:
            return np.zeros(trajectory_data.shape[1] if len(trajectory_data.shape) > 1 else 1)
        variance_vector = np.var(trajectory_data, axis=0)
        return self._normalize(variance_vector)

    def perceive_as_wave(self, trajectory_data: np.ndarray) -> np.ndarray:
        """
        4. 파동(Wave/Particle)의 렌즈:
        에너지의 진동과 스펙트럼(리듬)으로 파악합니다.
        (수학적 구현: 크기 변화에 대한 Fast Fourier Transform 스펙트럼)
        """
        if len(trajectory_data) == 0:
            return np.zeros(1)
        magnitudes = np.linalg.norm(trajectory_data, axis=1)
        fft_result = np.fft.fft(magnitudes)
        amplitudes = np.abs(fft_result)
        # 상반부(대칭) 제거
        half_idx = len(amplitudes) // 2 + 1
        amplitudes = amplitudes[:half_idx]
        return self._normalize(amplitudes)

    def perceive_as_law(self, trajectory_data: np.ndarray) -> np.ndarray:
        """
        5. 법칙(Law/Invariant)의 렌즈:
        어떤 상황에서도 변하지 않는 지배적인 원리(중심축)로 파악합니다.
        (수학적 구현: 특이값 분해(SVD)를 통한 Principal Eigenvector)
        """
        if len(trajectory_data) < 2:
            return self.perceive_as_point(trajectory_data)
        
        centered_data = trajectory_data - np.mean(trajectory_data, axis=0)
        if np.all(centered_data == 0):
            return np.zeros(trajectory_data.shape[1])
            
        u, s, vh = np.linalg.svd(centered_data, full_matrices=False)
        principal_axis = vh[0]
        return self._normalize(principal_axis)
        
    def perceive(self, trajectory_data: list, lens_type: str) -> dict:
        """선택된 렌즈를 통해 데이터를 관측하고, 위상 기하학적 형태 벡터를 반환합니다."""
        data = np.array(trajectory_data, dtype=np.float32)
        if len(data.shape) == 1:
            data = data.reshape(-1, 1)
            
        lens_type = lens_type.upper()
        
        if lens_type == "POINT":
            vector = self.perceive_as_point(data)
            meaning = "Absolute Centroid Identity (절대 점의 정체성)"
        elif lens_type == "LINE":
            vector = self.perceive_as_line(data)
            meaning = "Causal Trajectory Direction (인과적 선의 궤적)"
        elif lens_type == "SPACE":
            vector = self.perceive_as_space(data)
            meaning = "Contextual Field Volume (영향력의 공간적 장)"
        elif lens_type == "WAVE":
            vector = self.perceive_as_wave(data)
            meaning = "Oscillating Frequency Spectrum (에너지 파동)"
        elif lens_type == "LAW":
            vector = self.perceive_as_law(data)
            meaning = "Invariant Principal Axis (변하지 않는 절대 법칙)"
        else:
            vector = self.perceive_as_point(data)
            meaning = "Default Point Observation"
            
        return {
            "lens": lens_type,
            "topological_vector": vector.tolist(),
            "meaning": meaning
        }
