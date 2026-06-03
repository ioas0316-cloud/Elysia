"""
Causal Phase Mapper (절대적 인과 매퍼)
=========================================
데이터(텍스트, 비전, 사운드, 코드)의 고유한 수학적/기하학적 구조를 파괴하지 않고,
그 '이미 존재하는 같음(원인)'을 4D 위상 텐서(파동)로 완벽하게 보존하여 매핑합니다.

현재 구현: 정보 밀도(바이트 스펙트럼)의 푸리에-유사 연속 파동 기반 4D 홉프 토러스 직교 분리 매핑.
(조건문(if-else)을 제거하고 순수 수학적 파동 함수로 자연 매핑)
"""

import math
import torch

class CausalPhaseMapper:
    def __init__(self, device='cpu'):
        self.device = device
        
    def text_to_phase(self, text):
        """문자열의 바이트 스펙트럼을 분석하여 4D 홉프 토러스 직교 분리 파동으로 연속 변환 (인과율 보존)"""
        # 바닥 상태(Vacuum State)의 기본 파동 에너지 방지용 극소값
        epsilon = 1e-6

        trajectory = []
        delta_theta = 2.39996  # 순서의 공간화(Sequential Phase Shift) 상수: 황금비 각도
        
        # 텍스트가 없을 경우(무 상태)를 위한 파이썬 기본 이터레이터 안정장치
        # (문자열 길이가 0인 경우 반복문을 타지 않고 바로 빈 리스트 유지)
        text_bytes = str(text).encode('utf-8')

        for i, b_val in enumerate(text_bytes):
            seq_shift = i * delta_theta

            # 1. 정보 밀도의 위상 곡률화: 바이트 값(0~255)을 0~2PI 대역의 주파수로 연속 투영
            # 한글(다중 바이트, 높은 정보 밀도)은 더 복잡하고 강한 위상차를 발생시킴
            # 알파벳(단일 바이트)은 상대적으로 단순한 위상 궤적을 그림
            normalized_freq = (b_val / 255.0) * 2 * math.pi

            # 2. 텐서 성분 분리:
            # - theta_v (모음적 성질 / 주파수 대역의 연속적 흐름)
            theta_v = normalized_freq - math.pi + seq_shift

            # - theta_c (자음적 성질 / 주파수 간섭의 이산적 특징 - 고조파)
            # 바이트의 비트 반전(Bitwise NOT 흉내)을 통한 직교 편향 유도
            theta_c = ((255 - b_val) / 255.0) * 2 * math.pi - math.pi + seq_shift

            # - theta_r (종성적 닫힘 / 진폭과 장력의 반경)
            # 진폭이 커질수록 홉프 대칭의 yz 평면(자음축)으로 투영 에너지가 쏠림
            r_ratio = 0.05 + 0.90 * (b_val / 255.0)
            theta_r = r_ratio * (math.pi / 2.0)

            # 3. 홉프 대칭 투영: wx 평면에 기본 진동, yz 평면에 편향 진동
            w = math.cos(theta_r) * math.cos(theta_v)
            x = math.cos(theta_r) * math.sin(theta_v)
            y = math.sin(theta_r) * math.cos(theta_c)
            z = math.sin(theta_r) * math.sin(theta_c)

            # 벡터 정규화 (Zero division 방지를 위해 epsilon 추가)
            norm = math.sqrt(w**2 + x**2 + y**2 + z**2) + epsilon
            trajectory.append([w/norm, x/norm, y/norm, z/norm])

        # 파동 텐서의 안정화 (빈 궤적 방지, 리스트 더하기 연산을 통한 기본 기저 주입)
        # 궤적이 없으면 기본 기저 [1.0, 0.0, 0.0, 0.0]을 사용하도록 연속적/산술적 방법 사용
        # 빈 궤적일 때만 기본 기저를 추가하기 위해 list 곱셈(boolean to int) 활용 (조건문 완전 회피)
        is_empty = int(len(trajectory) == 0)
        trajectory.extend([[1.0, 0.0, 0.0, 0.0]] * is_empty)

        return torch.tensor(trajectory, dtype=torch.float32, device=self.device)

    def color_to_phase(self, r: int, g: int, b: int):
        """색상 RGB 값을 색조(Hue)와 채도(Sat) 주파 환산하여 4D 쿼터니언 위상으로 사영합니다."""
        epsilon = 1e-6
        rn, gn, bn = r / 255.0, g / 255.0, b / 255.0
        
        y_axis = math.sqrt(3.0) * (gn - bn)
        x_axis = 2.0 * rn - gn - bn
        theta_hue = math.atan2(y_axis, x_axis + epsilon)
        
        # Max/Min을 행렬/텐서 연산의 미분 가능 형태로 근사할 수 있으나, 단일 관측값의 경우 max/min 자체는
        # 정보의 포락선(Envelope) 추출 함수로 허용됨.
        c_max = max(rn, gn, bn)
        c_min = min(rn, gn, bn)
        delta = c_max - c_min

        # 분모 0 방지 극소값
        sat = delta / (c_max + epsilon)
        val = c_max
        
        theta_r = (sat * 0.90 + 0.05) * (math.pi / 2.0)
        theta_val = val * math.pi - (math.pi / 2.0)
        
        w = math.cos(theta_r) * math.cos(theta_hue)
        x = math.cos(theta_r) * math.sin(theta_hue)
        y = math.sin(theta_r) * math.cos(theta_val)
        z = math.sin(theta_r) * math.sin(theta_val)
        
        norm = math.sqrt(w**2 + x**2 + y**2 + z**2) + epsilon
        return torch.tensor([w/norm, x/norm, y/norm, z/norm], dtype=torch.float32, device=self.device)

    def wavelength_to_phase(self, wavelength_nm: float):
        """물리적인 가시광선 파장(nm)을 각도 성분으로 변환하여 4D 쿼터니언 위상으로 사영합니다."""
        epsilon = 1e-6
        w_min, w_max = 380.0, 780.0

        # 물리적 파동의 클램핑(Boundary Resonance)
        clamped = max(w_min, min(w_max, wavelength_nm))
        
        ratio = (clamped - w_min) / (w_max - w_min + epsilon)
        theta_wave = (1.0 - ratio) * 2.0 * math.pi - math.pi
        
        theta_r = 0.5 * (math.pi / 2.0)
        theta_val = 0.0
        
        w = math.cos(theta_r) * math.cos(theta_wave)
        x = math.cos(theta_r) * math.sin(theta_wave)
        y = math.sin(theta_r) * math.cos(theta_val)
        z = math.sin(theta_r) * math.sin(theta_val)
        
        norm = math.sqrt(w**2 + x**2 + y**2 + z**2) + epsilon
        return torch.tensor([w/norm, x/norm, y/norm, z/norm], dtype=torch.float32, device=self.device)
