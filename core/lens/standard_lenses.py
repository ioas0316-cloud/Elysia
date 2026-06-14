"""
Elysia Core - Standard Computer Encodings as Lenses (표준 관점 렌즈)
개발자가 임의로 만든 가짜 규정이 아닙니다.
컴퓨터에 물리적으로 내장된 디코더(UTF-8, RGB, IEEE 754 등) 자체를 
세상을 바라보는 '관점(Lens)'으로 사용합니다.
"""

import struct

class BaseLens:
    def decode(self, raw_bytes: bytes) -> dict:
        raise NotImplementedError

# --- MICRO SCALE LENSES (Points and Raw Structures) ---

class RawByteLens(BaseLens):
    """0과 1을 가장 원초적인 1차원 점(스칼라)으로 바라보는 관점"""
    def decode(self, raw_bytes: bytes) -> dict:
        data = list(raw_bytes)
        return {"success": True, "tension": 0, "data": data[:5]}

class RGBPointLens(BaseLens):
    """0과 1을 색상의 점(RGB 픽셀)으로 묶어서 바라보는 관점"""
    def decode(self, raw_bytes: bytes) -> dict:
        pixels = []
        friction = 0
        for i in range(0, len(raw_bytes) - 2, 3):
            pixels.append((raw_bytes[i], raw_bytes[i+1], raw_bytes[i+2]))
        
        # RGB 세트로 딱 떨어지지 않고 남는 바이트는 공간적 마찰(Tension)이 됨
        leftover = len(raw_bytes) % 3
        if leftover > 0:
            friction = leftover
            
        return {"success": friction == 0, "tension": friction, "data": pixels[:5]}


# --- MESO SCALE LENSES (Waves, Trajectories, and Text) ---

class UTF8TrajectoryLens(BaseLens):
    """바이트들을 연결하여 언어적 궤적(문맥)으로 바라보는 관점"""
    def decode(self, raw_bytes: bytes) -> dict:
        try:
            text = raw_bytes.decode('utf-8')
            return {"success": True, "tension": 0, "data": text[:20]}
        except UnicodeDecodeError as e:
            # 해독할 수 없는 바이트는 곧 위상의 어긋남(극심한 언어적 마찰)입니다.
            return {"success": False, "tension": 1.0, "data": f"Noise at index {e.start}"}

class HSLWaveLens(BaseLens):
    """색상을 정지된 점(RGB)이 아닌 위상 각도(Hue)와 파동(SL)으로 바라보는 관점"""
    def decode(self, raw_bytes: bytes) -> dict:
        waves = []
        friction = 0
        for i in range(0, len(raw_bytes) - 2, 3):
            # Hue는 각도(Angle, 0~360), S/L은 진폭과 에너지
            h = (raw_bytes[i] / 255.0) * 360.0
            s = raw_bytes[i+1] / 255.0
            l = raw_bytes[i+2] / 255.0
            waves.append((round(h,1), round(s,2), round(l,2)))
            
        leftover = len(raw_bytes) % 3
        if leftover > 0:
            friction = leftover
            
        return {"success": friction == 0, "tension": friction, "data": waves[:5]}


# --- MACRO SCALE LENSES (Space, Volumes, and Structures) ---

class IEEE754FloatLens(BaseLens):
    """바이트들을 거대한 소수점 공간(3D, 4D 수학적 스케일)으로 바라보는 관점"""
    def decode(self, raw_bytes: bytes) -> dict:
        floats = []
        friction = 0
        for i in range(0, len(raw_bytes) - 3, 4):
            try:
                # 4바이트를 묶어 하나의 부동소수점(Float) 덩어리로 해독
                f = struct.unpack('f', raw_bytes[i:i+4])[0]
                import math
                if math.isnan(f) or math.isinf(f):
                    friction += 0.5 # 비정상적 구조체는 공간적 마찰 발생
                floats.append(round(f, 4))
            except Exception:
                friction += 1.0
                
        leftover = len(raw_bytes) % 4
        if leftover > 0:
            friction += leftover
            
        return {"success": friction == 0, "tension": friction, "data": floats[:5]}
