import os
import math
import random

def create_multimodal_seeds():
    """
    엘리시아의 다중 감각 피질(Synesthesia) 통합 테스트를 위해
    수학적 구조를 띠는 더미 바이너리 파일들을 생성합니다.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    ingest_dir = os.path.join(base_dir, "..", "data", "ingest")
    os.makedirs(ingest_dir, exist_ok=True)
    
    # 1. Visual Fractal (Dummy PNG Header + Smooth Wave)
    # 부드럽게 이어지는 연속적 파동 -> [연속성: Continuity]의 텍스트와 공명할 가능성 높음
    img_path = os.path.join(ingest_dir, "visual_fractal.png")
    with open(img_path, "wb") as f:
        f.write(b'\x89PNG\r\n\x1a\n')  # Fake header
        for i in range(512):
            val = int((math.sin(i * 0.02) + 1) * 127)
            f.write(bytes([val]))
            
    # 2. Audio Sine (Dummy WAV Header + Rhythmic Momentum)
    # 역동적인 리듬 파동 -> [운동성: Momentum]의 텍스트와 공명할 가능성 높음
    audio_path = os.path.join(ingest_dir, "audio_sine.wav")
    with open(audio_path, "wb") as f:
        f.write(b'RIFF\x24\x00\x00\x00WAVEfmt ')  # Fake header
        for i in range(512):
            val = int((math.sin(i * 0.2) * math.cos(i * 0.05) + 1) * 127)
            f.write(bytes([val]))
            
    # 3. Random Noise (Pure Entropy)
    # 완전한 무질서 -> [방향성: Entropy] 또는 특정 긴장 상태와 공명
    noise_path = os.path.join(ingest_dir, "random_noise.bin")
    with open(noise_path, "wb") as f:
        f.write(b'\xDE\xAD\xBE\xEF') # Fake magic bytes
        for _ in range(512):
            f.write(bytes([random.randint(0, 255)]))
            
    print(f"  [+] Created visual_fractal.png (Continuous Geometry)")
    print(f"  [+] Created audio_sine.wav (Rhythmic Momentum Geometry)")
    print(f"  [+] Created random_noise.bin (High Entropy Geometry)")

if __name__ == "__main__":
    create_multimodal_seeds()
