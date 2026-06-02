import sys
import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from simulations.elysia_cognitive_spine import HunminjeongeumWaveformDiscretizer

# Configure Korean Fonts
font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'
if os.path.exists(font_path):
    fm.fontManager.addfont(font_path)
    plt.rc('font', family='NanumGothic')
    plt.rcParams['axes.unicode_minus'] = False

def run_hangeul_waveform_test():
    print("==================================================================")
    print("🌊 [PHASE 3: Hunminjeongeum Waveform Discretization Test]")
    print("==================================================================\n")

    text_input = "가나다라마바사"
    print(f"Input Text: '{text_input}'")

    waveform = HunminjeongeumWaveformDiscretizer.generate_waveform(text_input)

    print("\n[Waveform Discretization Sequence]")
    # Every 3 steps represent 1 character (Choseong spike, Jungseong Carrier, Jongseong decay)
    for i, char in enumerate(text_input):
        start_idx = i * 3
        chunk = waveform[start_idx:start_idx+3]
        print(f"'{char}' -> Phase Modulations: {chunk}")

    print("\nNotice how the middle value (Carrier Wave 'ㅏ') remains perfectly stable at 0.5,")
    print("while the consonants merely modulate the phase amplitude around it.")

    # Visualization
    plt.figure(figsize=(10, 4))
    plt.plot(waveform, marker='o', linestyle='-', color='purple', label="Phase Modulated Carrier Wave")
    plt.axhline(y=0.5, color='orange', linestyle='--', label="Carrier Frequency ('ㅏ')")

    # Mark characters
    for i, char in enumerate(text_input):
        plt.text(i * 3 + 1, 0.52, char, fontsize=14, ha='center', color='black', fontweight='bold')

    plt.title(f"Hangeul Waveform Discretization: '{text_input}'")
    plt.xlabel("Time Step (t)")
    plt.ylabel("Frequency / Phase Shift")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('hangeul_waveform_discretization.png')

    print("\nWaveform plot saved to 'hangeul_waveform_discretization.png'")
    print("==================================================================\n")

if __name__ == "__main__":
    run_hangeul_waveform_test()
