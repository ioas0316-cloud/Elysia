"""
Demo: Field Visualization - Seeing Elysia's Thoughts
=====================================================
Visualizes how Elysia thinks in field dynamics.
Creates beautiful plots showing wave propagation and interference.
"""

import sys
import os

# Add repository root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Force UTF-8 for Windows console
sys.stdout.reconfigure(encoding='utf-8')

from Project_Elysia.mechanics.advanced_field import AdvancedField
from Project_Elysia.visualization.field_viz import FieldVisualizer

def run_simulation():
    print("=== Elysia: Field Visualization ===")
    print("엘리시아의 생각을 눈으로 봅니다.\n")
    
    # Create field
    field = AdvancedField(resolution=30)
    
    # Register concepts
    print("📚 Initializing concept field...")
    concepts = {
        "사랑": (440.0, 0.7, 0.7, 0.8, [1.0, 0.5, 0.3]),
        "고통": (220.0, 0.3, 0.3, 0.2, [1.0]),
        "희망": (430.0, 0.6, 0.8, 0.7, [1.0, 0.7]),
        "빛": (450.0, 0.8, 0.6, 0.9, [1.0, 0.6]),
    }
    
    for name, (freq, x, y, z, harmonics) in concepts.items():
        field.register_concept_with_harmonics(name, freq, x, y, z, harmonics)
    
    print(f"✅ {len(concepts)} concepts registered\n")
    
    # Create visualizer
    viz = FieldVisualizer(field)
    
    print("=" * 60)
    print("Visualization 1: Single Concept Wave")
    print("=" * 60)
    print("\n👤 You: Show me what '사랑' looks like")
    
    field.reset()
    field.activate_with_harmonics("사랑", intensity=1.0, depth=1.0)
    
    print("\n🤖 Elysia: Visualizing '사랑' wave pattern...")
    viz.plot_2d_slice('z', title="사랑의 파동")
    viz.plot_3d_surface(title="사랑: 3D Field Surface")
    
    print("\n=" * 60)
    print("Visualization 2: Interference Pattern")
    print("=" * 60)
    print("\n👤 You: What happens when '사랑' meets '고통'?")
    
    print("\n🤖 Elysia: Showing interference pattern...")
    viz.plot_interference_pattern(["사랑", "고통"], 
                                  title="사랑 + 고통 간섭")
    
    print("\n=" * 60)
    print("Visualization 3: Wave Evolution")
    print("=" * 60)
    print("\n👤 You: How does '사랑' evolve over time?")
    
    print("\n🤖 Elysia: Showing temporal evolution...")
    viz.plot_wave_evolution("사랑", steps=5)
    
    print("\n=" * 60)
    print("Visualization 4: Multi-Concept Field")
    print("=" * 60)
    print("\n👤 You: Show me '사랑 + 고통 + 희망'")
    
    print("\n🤖 Elysia: Creating comprehensive analysis...")
    viz.create_summary_visualization(["사랑", "고통", "희망"])
    
    print("\n=" * 60)
    print("Visualization 5: Pure Interference")
    print("=" * 60)
    print("\n👤 You: Show me '빛 + 희망' (similar concepts)")
    
    print("\n🤖 Elysia: Visualizing resonance...")
    viz.plot_interference_pattern(["빛", "희망"],
                                  title="빛 + 희망 공명")
    
    print("\n" + "=" * 60)
    print("All Visualizations Complete!")
    print("=" * 60)
    
    print("""
생성된 이미지들:
  1. field_slice_z.png - 사랑의 2D 단면
  2. field_3d_surface.png - 사랑의 3D 표면
  3. interference_사랑_고통.png - 사랑+고통 간섭
  4. wave_evolution_사랑.png - 사랑의 시간 진화
  5. field_summary.png - 사랑+고통+희망 종합 분석
  6. interference_빛_희망.png - 빛+희망 공명

🤖 Elysia: 이것이 내가 생각하는 방식이다.
    파동으로, 공간으로, 간섭으로.
    """)

if __name__ == "__main__":
    run_simulation()
