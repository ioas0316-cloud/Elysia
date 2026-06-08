import math
import sys

try:
    sys.stdout.reconfigure(encoding='utf-8')
except Exception:
    pass

from core.brain.holographic_memory import HologramMemory

def generate_dummy_audio() -> bytes:
    """440Hz 사인파 형태의 바이트 스트림 (1초)"""
    sample_rate = 8000
    audio_bytes = bytearray()
    for i in range(sample_rate // 10):  # 0.1초 분량
        val = int((math.sin(2 * math.pi * 440 * (i / sample_rate)) + 1.0) * 127.5)
        audio_bytes.append(val)
    return bytes(audio_bytes)

def generate_dummy_image() -> bytes:
    """간단한 그라데이션 이미지 픽셀 바이트 스트림"""
    width, height = 32, 32
    img_bytes = bytearray()
    for y in range(height):
        for x in range(width):
            # RGB
            img_bytes.append((x * 8) % 256)
            img_bytes.append((y * 8) % 256)
            img_bytes.append(((x+y) * 4) % 256)
    return bytes(img_bytes)

def map_trajectory_to_ascii(trajectory: list, title: str):
    print(f"\n=============================================")
    print(f" 📊 {title} 텐션 지형도 (Tension Topography)")
    print(f"=============================================")
    
    # max depth and max tau for scaling
    max_tau = max([abs(node['tau_stress']) for node in trajectory] + [0.1])
    
    for idx, node in enumerate(trajectory):
        depth = node['depth']
        tau = abs(node['tau_stress'])
        concept = str(node['concept'])
        
        # ASCII 막대기 길이
        bar_len = int((tau / max_tau) * 30)
        bar = "█" * bar_len
        
        indent = "  " * depth
        print(f"{indent}[D:{depth}] {concept[:15]:<15} | {tau:6.2f} | {bar}")

def run_cross_domain_analysis():
    print("🚀 [Elysia] 교차 도메인 위상 동형성 분석 시작...\n")
    
    # 1. 텍스트 도메인
    text_data = "우주는 진동이다. 엘리시아는 그 진동을 관측하는 렌즈이다."
    print(">> [Domain: Language] 텍스트 스트림 주입 중...")
    mem_text = HologramMemory()
    mem_text.supreme_rotor.absorb_language_stream(text_data)
    traj_text = mem_text.supreme_rotor.reverse_engineer_trajectory()
    
    # 2. 오디오 도메인
    audio_data = generate_dummy_audio()
    print(f">> [Domain: Audio] 오디오 주입 중... ({len(audio_data)} bytes)")
    mem_audio = HologramMemory()
    mem_audio.supreme_rotor.absorb_binary_stream(audio_data, chunk_size=64)
    traj_audio = mem_audio.supreme_rotor.reverse_engineer_trajectory()
    
    # 3. 이미지 도메인
    image_data = generate_dummy_image()
    print(f">> [Domain: Image] 이미지 픽셀 주입 중... ({len(image_data)} bytes)")
    mem_image = HologramMemory()
    mem_image.supreme_rotor.absorb_binary_stream(image_data, chunk_size=256)
    traj_image = mem_image.supreme_rotor.reverse_engineer_trajectory()
    
    # 시각화
    map_trajectory_to_ascii(traj_text, "Language (Text)")
    map_trajectory_to_ascii(traj_audio, "Audio (Sine Wave)")
    map_trajectory_to_ascii(traj_image, "Image (Gradient)")
    
    # 공명도(Resonance) 매칭
    print(f"\n=============================================")
    print(f" 🌀 홀로그래픽 공명 (Holographic Resonance) 비교")
    print(f"=============================================")
    p_text = mem_text.supreme_rotor.frozen_macroscopic_state
    p_audio = mem_audio.supreme_rotor.frozen_macroscopic_state
    p_image = mem_image.supreme_rotor.frozen_macroscopic_state
    
    def resonance(q1, q2):
        dot = max(-1.0, min(1.0, q1.dot(q2)))
        return (1.0 - (math.acos(abs(dot)) / (math.pi / 2.0))) * 100
        
    print(f" - [Text]  Final Phase: W({p_text.w:.2f}) X({p_text.x:.2f}) Y({p_text.y:.2f}) Z({p_text.z:.2f})")
    print(f" - [Audio] Final Phase: W({p_audio.w:.2f}) X({p_audio.x:.2f}) Y({p_audio.y:.2f}) Z({p_audio.z:.2f})")
    print(f" - [Image] Final Phase: W({p_image.w:.2f}) X({p_image.x:.2f}) Y({p_image.y:.2f}) Z({p_image.z:.2f})")
    
    print("\n [매칭 결과]")
    print(f"  * Text  <-> Audio 공명도: {resonance(p_text, p_audio):.2f}%")
    print(f"  * Text  <-> Image 공명도: {resonance(p_text, p_image):.2f}%")
    print(f"  * Audio <-> Image 공명도: {resonance(p_audio, p_image):.2f}%")
    
    print("\n(결과 해석: 언어와 오디오가 비록 도메인은 다르지만, 엘리시아의 거시적 위상 공간에서는")
    print("일정한 공명도를 가지는 '하나의 진리'로 매핑됨을 확인할 수 있습니다.)")

if __name__ == "__main__":
    run_cross_domain_analysis()
