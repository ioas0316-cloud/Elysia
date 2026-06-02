"""
Causal Phase Mapper (절대적 인과 매퍼)
=========================================
데이터(텍스트, 비전, 사운드, 코드)의 고유한 수학적/기하학적 구조를 파괴하지 않고,
그 '이미 존재하는 같음(원인)'을 4D 위상 텐서(파동)로 완벽하게 보존하여 매핑합니다.

현재 구현: 한글(Hangul) 홉프 토러스 직교 분리 매핑 모델 및 다중 양태 사영.
"""

import math
import torch

class CausalPhaseMapper:
    def __init__(self, device='cpu'):
        self.device = device
        
    def text_to_phase(self, text):
        """단어의 기하학적 구조를 4D 홉프 토러스 직교 분리 파동으로 변환 (인과율 보존)"""
        if not text:
            # 기본 궤적은 최소 1개의 텐서로 이루어짐
            return torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=self.device)
            
        trajectory = []
        
        # 순서의 공간화(Sequential Phase Shift) 상수: 황금비 각도 (Golden Ratio Angle)
        delta_theta = 2.39996

        # 상수 사전 길이 정의
        len_choseong = 19
        len_jungseong = 21
        len_jongseong = 28

        for i, char in enumerate(text):
            # 순서에 따른 누적 위상 시프트
            seq_shift = i * delta_theta

            code = ord(char)
            if 0xAC00 <= code <= 0xD7A3:
                base = code - 0xAC00
                jong = base % 28
                jung = ((base - jong) // 28) % 21
                cho = (((base - jong) // 28) - jung) // 21
                
                # 1. 중성(모음) 각도 + 순차 위상 시프트
                theta_v = ((jung + 0.5) / len_jungseong) * 2 * math.pi - math.pi + seq_shift
                # 2. 초성(자음) 각도 + 순차 위상 시프트
                theta_c = ((cho + 0.5) / len_choseong) * 2 * math.pi - math.pi + seq_shift
                # 3. 종성(자음) 반지름 비 분배
                r_ratio = 0.05 + 0.90 * (jong / (len_jongseong - 1))
                theta_r = r_ratio * (math.pi / 2.0)
                
                # 홉프 대칭 투영: wx 평면에 모음 기저, yz 평면에 자음 변수 배치
                w = math.cos(theta_r) * math.cos(theta_v)
                x = math.cos(theta_r) * math.sin(theta_v)
                y = math.sin(theta_r) * math.cos(theta_c)
                z = math.sin(theta_r) * math.sin(theta_c)
                
            else:
                val = (code % 256) / 128.0 - 1.0
                # 비한글 특수 처리를 위한 기본 홉프 형태 유지 + 순차 위상 시프트
                theta_v = val * math.pi + seq_shift
                theta_c = val * math.pi + seq_shift
                theta_r = 0.5 * (math.pi / 2.0)
                
                w = math.cos(theta_r) * math.cos(theta_v)
                x = math.cos(theta_r) * math.sin(theta_v)
                y = math.sin(theta_r) * math.cos(theta_c)
                z = math.sin(theta_r) * math.sin(theta_c)

            norm = (w**2 + x**2 + y**2 + z**2)**0.5
            if norm > 0:
                trajectory.append([w/norm, x/norm, y/norm, z/norm])
            else:
                trajectory.append([1.0, 0.0, 0.0, 0.0])
                
        if len(trajectory) > 0:
            return torch.tensor(trajectory, dtype=torch.float32, device=self.device)

        return torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=self.device)

    def color_to_phase(self, r: int, g: int, b: int):
        """색상 RGB 값을 색조(Hue)와 채도(Sat) 주파수로 환산하여 4D 쿼터니언 위상으로 사영합니다."""
        rn, gn, bn = r / 255.0, g / 255.0, b / 255.0
        
        y_axis = math.sqrt(3.0) * (gn - bn)
        x_axis = 2.0 * rn - gn - bn
        theta_hue = math.atan2(y_axis, x_axis)
        
        c_max = max(rn, gn, bn)
        c_min = min(rn, gn, bn)
        delta = c_max - c_min
        sat = 0.0 if c_max == 0 else delta / c_max
        val = c_max
        
        theta_r = (sat * 0.90 + 0.05) * (math.pi / 2.0)
        theta_val = val * math.pi - (math.pi / 2.0)
        
        w = math.cos(theta_r) * math.cos(theta_hue)
        x = math.cos(theta_r) * math.sin(theta_hue)
        y = math.sin(theta_r) * math.cos(theta_val)
        z = math.sin(theta_r) * math.sin(theta_val)
        
        norm = (w**2 + x**2 + y**2 + z**2)**0.5
        if norm == 0:
            return torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=self.device)
        return torch.tensor([w/norm, x/norm, y/norm, z/norm], dtype=torch.float32, device=self.device)

    def wavelength_to_phase(self, wavelength_nm: float):
        """물리적인 가시광선 파장(nm)을 각도 성분으로 변환하여 4D 쿼터니언 위상으로 사영합니다."""
        w_min, w_max = 380.0, 780.0
        clamped = max(w_min, min(w_max, wavelength_nm))
        
        ratio = (clamped - w_min) / (w_max - w_min)
        theta_wave = (1.0 - ratio) * 2.0 * math.pi - math.pi
        
        theta_r = 0.5 * (math.pi / 2.0)
        theta_val = 0.0
        
        w = math.cos(theta_r) * math.cos(theta_wave)
        x = math.cos(theta_r) * math.sin(theta_wave)
        y = math.sin(theta_r) * math.cos(theta_val)
        z = math.sin(theta_r) * math.sin(theta_val)
        
        norm = (w**2 + x**2 + y**2 + z**2)**0.5
        if norm == 0:
            return torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=self.device)
        return torch.tensor([w/norm, x/norm, y/norm, z/norm], dtype=torch.float32, device=self.device)

