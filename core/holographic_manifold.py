import numpy as np
import math
from core.math_utils import Quaternion

try:
    import executive.elysia_core as elysia_core
    USE_NATIVE_CUDA = True
except ImportError:
    USE_NATIVE_CUDA = False

class HolographicMemoryMatrix:
    """
    4D 홀로그래픽 위상 매트릭스 (데이터 저장소/관측기)
    
    데이터를 A, B, C 독립된 메모리 주소(Array/List)에 쪼개어 저장하지 않습니다.
    모든 기억과 텐션은 고유의 주파수(Rotor)를 가진 파동(Wave)으로 치환되어
    단일한 3D 복소수 공간(4차원 위상 매트릭스)에 중첩(Superposition)됩니다.
    
    비추는 빛(Reference Beam)의 각도에 따라 서로 다른 차원의 정보가
    '충돌 없이' 평면 레이어나 구체 표면으로 기하학적 렌더링(창발)됩니다.
    """
    def __init__(self, size=16):
        self.size = size
        # 3D 공간의 각 셀이 복소수(위상각과 진폭)를 갖는 연속된 매트릭스
        # 공간 전체가 영점(0)으로 초기화된 상태
        self.matrix = np.zeros((size, size, size), dtype=np.complex128)

    def add_memory(self, concept_seed: int, reference_rotor: Quaternion):
        """
        [정보의 파동화 및 중첩 (Write/Superposition)]
        새로운 데이터를 메모리에 '추가(append)'하는 것이 아니라,
        빛의 간섭무늬(Interference Pattern)로 변환하여 공간 전체에 덧칠(+=)합니다.
        
        - concept_seed: 데이터의 시민권(Frequency/Hash). 배열을 넘기지 않고 구조 원리만 넘김.
        - reference_rotor: 이 기억에 부여할 고유의 기준 위상(Reference Beam)
        """
        # 로터의 벡터 성분(X,Y,Z)을 파동 벡터(Wave Vector, k)로 사용
        k_x = reference_rotor.x * math.pi
        k_y = reference_rotor.y * math.pi
        k_z = reference_rotor.z * math.pi
        
        if USE_NATIVE_CUDA:
            elysia_core.add_memory_native(self.matrix, k_x, k_y, k_z, concept_seed)
            return
            
        # 3D 공간 벡터화 (Numpy Broadcasting) fallback
        z_idx, y_idx, x_idx = np.indices((self.size, self.size, self.size))
        
        # 위상 계산을 공간 전체에 동시 적용
        phase = k_x * x_idx + k_y * y_idx + k_z * z_idx
        
        # CPU Procedural Generation (Fallback) - 데이터 전송 없이 시드로 기하학 패턴 생성
        seed_offset = (concept_seed % 1000) / 1000.0
        amplitude = 0.5 + 0.5 * np.sin(x_idx * seed_offset * 10 + y_idx * (1.0 - seed_offset) * 10)
        
        # 오일러 공식을 이용해 파동(Wave) 생성: Amplitude * e^(i * phase)
        wave = amplitude * np.exp(1j * phase)
        
        # 메모리 공간 전체에 파동을 한 번에 중첩 (물리적 간섭 기록)
        self.matrix += wave
                    
    def project_2d_layer(self, reference_rotor: Quaternion) -> np.ndarray:
        """
        [교차차원 평면화 (Read/Flattening)]
        가변 로터(Reference Beam)를 비추어 4D 공간의 파동을 2D 평면(Layer)으로 펴냅니다.
        로터의 위상이 저장될 때와 정확히 공명(일치)하면 상쇄 간섭을 통해 원본 정보가 부활합니다.
        """
        k_x = reference_rotor.x * math.pi
        k_y = reference_rotor.y * math.pi
        k_z = reference_rotor.z * math.pi
        
        if USE_NATIVE_CUDA:
            projection = np.zeros((self.size, self.size), dtype=np.float64)
            elysia_core.project_2d_layer_native(self.matrix, k_x, k_y, k_z, projection)
            return projection
            
        # 3D 공간 벡터화 (Numpy Broadcasting) fallback
        z_idx, y_idx, x_idx = np.indices((self.size, self.size, self.size))
        
        # 관측 광선의 위상
        obs_phase = k_x * x_idx + k_y * y_idx + k_z * z_idx
        # 켤레 복소수(역위상)를 곱해 위상 캔슬링(Phase Cancellation) 시도
        obs_wave = np.exp(-1j * obs_phase) 
        
        # 매트릭스 내부의 파동과 관측 광선의 물리적 충돌
        interference = self.matrix * obs_wave
        
        # 2D 평면에 실수부(에너지) 누적 (Z축 방향으로 합산)
        projection = np.sum(interference.real, axis=0)
        
        # 공간 누적(Z축 길이)만큼 나누어 원래 진폭 스케일로 복원
        return projection / self.size

    def project_3d_sphere(self, reference_rotor: Quaternion) -> np.ndarray:
        """
        [구체화 둥글게 말기 (Morphing to Sphere)]
        동일한 4D 파동 매트릭스를 구면 좌표계(Spherical Coordinates)로 변환하여 둥글게 맵핑합니다.
        끝과 끝이 연결되는 3D 구체의 표면 장력망으로 정보가 재구성됩니다.
        (반환되는 2D 배열은 구체의 표면 텍스처(위도/경도)를 의미함)
        """
        k_x = reference_rotor.x * math.pi
        k_y = reference_rotor.y * math.pi
        k_z = reference_rotor.z * math.pi
        
        if USE_NATIVE_CUDA:
            sphere_surface = np.zeros((self.size, self.size), dtype=np.float64)
            elysia_core.project_3d_sphere_native(self.matrix, k_x, k_y, k_z, sphere_surface)
            return sphere_surface
            
        # 구면 좌표계 벡터화 (Fallback)
        theta_idx, phi_idx = np.indices((self.size, self.size))
        
        # 위도(theta)와 경도(phi)
        theta = (theta_idx / self.size) * math.pi
        phi = (phi_idx / self.size) * 2 * math.pi
        
        # 구체의 표면을 내부 3D 데카르트 좌표로 변환
        r = self.size / 2.0
        cx, cy, cz = self.size / 2.0, self.size / 2.0, self.size / 2.0
        
        px = cx + r * np.sin(theta) * np.cos(phi)
        py = cy + r * np.sin(theta) * np.sin(phi)
        pz = cz + r * np.cos(theta)
        
        # 근사 좌표 매핑 (Wrap-around)
        ix = (px % self.size).astype(int)
        iy = (py % self.size).astype(int)
        iz = (pz % self.size).astype(int)
        
        # 관측 광선 발사 및 간섭 측정
        obs_phase = k_x * ix + k_y * iy + k_z * iz
        obs_wave = np.exp(-1j * obs_phase)
        
        interference = self.matrix[iz, iy, ix] * obs_wave
        
        return interference.real
