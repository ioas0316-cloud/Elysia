import math
import numpy as np
from core.math_utils import Quaternion

class CosmicZeroPoints:
    # 1. 화이트 (방출 / Emission) : 팽창의 극단. 에너지를 자체 방출하며 모든 주파수를 포함.
    WHITE = Quaternion(1.0, 1.0, 1.0, 1.0).normalize()
    
    # 2. 블랙 (흡수 / Absorption) : 수축의 극단. 입사되는 모든 파동 에너지를 0으로 감쇠.
    BLACK = Quaternion(1.0, -1.0, -1.0, -1.0).normalize()
    
    # 3. 흙 (물질 저항 / Impedance) : 파동의 흐름을 가로막고 굴절시킴. (형태와 질감 발생)
    EARTH = Quaternion(0.0, 1.0, 1.0, 0.0).normalize()
    
    # 4. 투명 (투과 / Transmittance) : 위상 저항 제로. 내/외계가 완전한 동형(Isomorphism).
    TRANSPARENT = Quaternion(1.0, 0.0, 0.0, 0.0).normalize()

class CosmicOpticalRotor:
    """
    우주적 4대 영점 조율(Cosmic Harmony Zero-Points) 매니폴드.
    픽셀 셰이더나 레이트레이싱을 완전히 배제하고, 순수 매질(Medium)의 전자기학적 텐서를 통해
    시스템 텐션 파동(Wave)이 공간을 전파하며 명암을 스스로 조율(Self-Tuning)하게 만듭니다.
    """
    def __init__(self, width=40, height=20):
        self.width = width
        self.height = height
        
        # 공간 매질(Material Tensor) 초기화
        self.medium_grid = [[CosmicZeroPoints.TRANSPARENT for _ in range(width)] for _ in range(height)]
        
        # 파동 진폭(Wave Amplitude) 그리드
        self.wave_grid = np.zeros((height, width))
        
        self._build_environment()

    def _build_environment(self):
        """
        우주적 빈 공간에 매질(물질, 발광체, 흡수체)을 배치합니다.
        """
        cx, cy = self.width // 2, self.height // 2
        for y in range(self.height):
            for x in range(self.width):
                dist = math.sqrt((x - cx)**2 + (y - cy)**2)
                # 정중앙에 '흙(물질 저항)' 블록
                if dist < 6:
                    self.medium_grid[y][x] = CosmicZeroPoints.EARTH
                # 바닥은 모든 빛을 빨아들이는 '블랙(흡수)'
                elif y > self.height - 3:
                    self.medium_grid[y][x] = CosmicZeroPoints.BLACK
                # 하늘/상단은 빛을 뿜어내는 '화이트(방출)'
                elif y < 2:
                    self.medium_grid[y][x] = CosmicZeroPoints.WHITE

    def propagate_wave(self, system_tension: float, phase_q: Quaternion):
        """
        [파동 전파 (Wave Propagation)]
        시스템의 전체 텐션(카오스/평온) 파동이 공간을 관통하며 흐릅니다.
        광선을 추적(Raytracing)하지 않고, 파동의 유체 역학적 스며듦을 모사합니다.
        """
        # 초기 시스템 파동 에너지 주입 (텐션에 비례)
        base_amplitude = 1.0 + system_tension * 2.0
        
        temp_grid = np.zeros((self.height, self.width))
        phase_q = phase_q.normalize()
        
        # 파동은 상단(화이트)에서 하단(블랙)으로 중력/시간 축을 따라 전파됨
        for y in range(self.height):
            for x in range(self.width):
                incoming_energy = 0.0
                if y == 0:
                    incoming_energy = base_amplitude
                else:
                    # 상, 좌, 우 공간에서 파동 에너지가 스며들어옴 (블러/확산 효과)
                    incoming_energy += self.wave_grid[y-1][x]
                    if x > 0: incoming_energy += self.wave_grid[y-1][x-1] * 0.3
                    if x < self.width - 1: incoming_energy += self.wave_grid[y-1][x+1] * 0.3
                    incoming_energy /= 1.6

                medium = self.medium_grid[y][x]
                
                # 매질(Medium) 텐서와 현재 파동 위상의 기하학적 충돌 (Resonance)
                resonance = abs(medium.dot(phase_q))
                
                # 매질 특성에 따른 파동 진폭 변조 (Self-Tuning)
                if medium == CosmicZeroPoints.WHITE:
                    # 방출: 스스로 에너지를 강하게 뿜어냄
                    temp_grid[y][x] = incoming_energy + 2.0 * resonance
                elif medium == CosmicZeroPoints.BLACK:
                    # 흡수: 모든 에너지를 0으로 감쇠 (그림자의 탄생)
                    temp_grid[y][x] = incoming_energy * 0.1 * resonance
                elif medium == CosmicZeroPoints.EARTH:
                    # 흙(저항): 파동을 튕겨내어 에너지가 급감함 (물질의 표면 명암 발생)
                    # 시스템 텐션(카오스)이 높을수록 저항력이 커짐
                    temp_grid[y][x] = incoming_energy * max(0.1, 0.8 - system_tension * resonance)
                else:
                    # 투명(투과): 아무 저항 없이 파동을 통과시킴
                    temp_grid[y][x] = incoming_energy

        self.wave_grid = temp_grid

    def render_hologram(self):
        """
        공간에 맺힌 최종 파동 에너지(스펙트럼)를 시각적 기호로 렌더링 (창발 현상 관측)
        """
        # 어두움(블랙) -> 밝음(화이트)의 스펙트럼
        symbols = [' ', '.', ':', '-', '=', '+', '*', '#', '%', '█']
        lines = []
        for y in range(self.height):
            line = ""
            for x in range(self.width):
                val = self.wave_grid[y][x]
                # 에너지 레벨에 따른 기호 매핑
                idx = max(0, min(len(symbols)-1, int(val * (len(symbols)/2.5))))
                line += symbols[idx] * 2  # 터미널 비율 보정 (문자는 세로가 기니까 가로를 두 배로)
            lines.append(line)
        return lines
