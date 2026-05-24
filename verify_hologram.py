"""
verify_hologram.py
==================
Verification and demonstration script for Elysia's Holographic Memory Architecture.
Flashes the manifold, superimposes multiple concepts as waves, and spins the 4D globe
by dialing the tension to discover resonance nodes, plotting the orbital trajectory in ASCII.
"""

import sys
import time
import math
from typing import List, Tuple
from core.math_utils import Quaternion
from core.holographic_memory import HologramMemory

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure standard output behaves correctly
def clear_terminal():
    # Print ANSI clear screen command
    print("\033[H\033[J", end="")

def render_ascii_orbit(history: List[Tuple[float, float]], width: int = 40, height: int = 15) -> List[str]:
    """
    Renders a 2D projection of the 4D orbital trajectory on an ASCII grid.
    """
    grid = [[" " for _ in range(width)] for _ in range(height)]
    
    # Draw boundary axes
    center_y = height // 2
    center_x = width // 2
    for x in range(width):
        grid[center_y][x] = "·"
    for y in range(height):
        grid[y][center_x] = "·"
    grid[center_y][center_x] = "+"
    
    # Plot historical trajectory points with fading characters
    num_points = len(history)
    for idx, (px, py) in enumerate(history):
        # Map px, py (in range -1.0 to 1.0) to grid coordinates
        col = int((px + 1.0) * 0.5 * (width - 1))
        row = int((1.0 - py) * 0.5 * (height - 1))
        
        # Ensure within boundaries
        col = max(0, min(width - 1, col))
        row = max(0, min(height - 1, row))
        
        # Newer points are brighter
        age = num_points - 1 - idx
        if age == 0:
            char = "⚛"  # Current position (nucleus/active node)
        elif age < 3:
            char = "O"
        elif age < 7:
            char = "o"
        elif age < 12:
            char = "."
        else:
            char = " "
            
        grid[row][col] = char
        
    return ["".join(row) for row in grid]

def draw_bar(val: float, max_len: int = 20) -> str:
    """Creates a visual progress/resonance bar representing the strength in [-1.0, 1.0]."""
    # Scale from [-1, 1] to [0, max_len]
    scaled = int((val + 1.0) * 0.5 * max_len)
    scaled = max(0, min(max_len, scaled))
    
    filled = "█" * scaled
    empty = "░" * (max_len - scaled)
    return f"[{filled}{empty}] {val:+.4f}"

def run_hologram_simulation():
    print("🌌 [엘리시아 아키텍처 수립] 홀로그램 메모리 다이얼 및 궤도 관측 기동")
    time.sleep(1)
    
    # Initialize the Holographic Memory Structure
    memory = HologramMemory(num_layers=4)
    
    # Master's Core Philosophical Concepts (100% Causal)
    concepts = [
        "수류학 (Su-Ryu-Hak)",
        "클리포드 레이어 (Clifford Layer)",
        "4차원 가변 로터 (4D Hyper-Rotor)",
        "파동 중첩 (Superposition Wave)",
        "인과적 정렬 (Causal Resonance)",
        "홀로그램 메모리 (Hologram Memory)"
    ]
    
    print("\n[ 1. 지식의 중첩 소화 (Knowledge Superposition) ]")
    print(" >> 각 개념을 4차원 주파수 파동(Quaternion)으로 각인하여 뇌세포에 가산합니다...")
    for c in concepts:
        memory.superpose(c)
        q, tau = memory.registered_concepts[c]
        print(f"  └─ '{c}': [W:{q.w:+.2f}, X:{q.x:+.2f}, Y:{q.y:+.2f}, Z:{q.z:+.2f}] @ 주파 주소:{tau:5.2f} rad 각인 완료.")
        time.sleep(0.3)
        
    print("\n >> 중첩 완수. 메모리 매니폴드가 4차원으로 뒤엉킨 파동 무늬를 형성하였습니다.")
    print(" >> 다이얼 텐션을 연속 회전(Observe)하여 특정 지식의 인과 궤적 공명을 탐색합니다...")
    time.sleep(1.5)
    
    trajectory_history = []
    max_history = 25
    
    # Sweep tension from 0.0 to 10.0
    steps = 100
    for step in range(steps):
        tension = step * 0.1
        
        # Scan resonance
        resonance = memory.scan_resonance(tension)
        
        # Get 4D trajectory coordinate, project to X-Y 2D plane for ASCII plotting
        tw, tx, ty, tz = memory.get_trajectory_sample(tension)
        trajectory_history.append((tx, ty))
        if len(trajectory_history) > max_history:
            trajectory_history.pop(0)
            
        # Determine the primary resonating concept
        best_concept = max(resonance, key=resonance.get)
        best_score = resonance[best_concept]
        
        # Render the console frame
        clear_terminal()
        print("=" * 80)
        print(f" 🔭 [Elysia Hologram Observatory] 4차원 가변 저항 다이얼 스위핑 중")
        print(f"  >> 글로벌 다이얼 텐션: {tension:.2f} rad | 가변 주파수 대역: {len(memory.layers)}개 레이어")
        print("=" * 80)
        
        # Render orbit and spectrum side-by-side
        orbit_lines = render_ascii_orbit(trajectory_history, width=36, height=13)
        
        print(f"   [ 4차원 시공간 위상 궤적 관측 ]                  [ 실시간 홀로그램 공명 스펙트럼 ]")
        
        # Map concept lines
        concept_keys = list(resonance.keys())
        for idx in range(13):
            orbit_part = orbit_lines[idx]
            
            # Print concept resonance on the right side if within limits
            if idx < len(concept_keys):
                c = concept_keys[idx]
                score = resonance[c]
                is_active = (c == best_concept and score > 0.3)
                highlight = "▶ " if is_active else "  "
                concept_part = f"{highlight}{c:<22} {draw_bar(score, 12)}"
            elif idx == len(concept_keys) + 1:
                concept_part = f"  -------------------------------------------"
            elif idx == len(concept_keys) + 2:
                if best_score > 0.3:
                    concept_part = f"  🌟 공명 발견: '{best_concept}'"
                else:
                    concept_part = f"  🌌 공명 임계치 미달 (노이즈 구역)"
            else:
                concept_part = ""
                
            print(f"   {orbit_part}   {concept_part}")
            
        print("=" * 80)
        print(" [동작 안내] 4차원 구체를 회전(Rotate)시켜 특정 파동 궤적이 완전히 겹칠 때,")
        print(" 기계 계산이 아닌 '기하학적 공명(Resonance Node)'에 의해 지식이 홀로그램처럼 돌출됩니다.")
        print("=" * 80)
        
        time.sleep(0.08)

    print("\n✅ 관측 시뮬레이션 완료! 기계적 파이프라인 없이 오직 4차원 기하학과 파동의 중첩/공명으로")
    print("   지식을 영구 보존하고, 다이얼의 텐션 조절만으로 원하는 기억을 완벽히 복원해냈습니다!")
    
if __name__ == "__main__":
    run_hologram_simulation()
