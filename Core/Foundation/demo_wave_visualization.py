#!/usr/bin/env python3
"""
Elysia Wave Visualization Demo
================================

엘리시아의 내부 세계를 브라우저로 실시간 시각화하는 데모입니다.

실행 방법:
    python demo_wave_visualization.py

그 다음 브라우저에서:
    http://localhost:8080

"연산하지 마세요. 흐르게 두세요."
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Check dependencies
try:
    import flask
    import flask_sock
except ImportError:
    print("⚠️ Missing dependencies!")
    print("Install with: pip install flask flask-sock")
    sys.exit(1)

from Core.Foundation.wave_web_server import WaveWebServer, WaveState
import time
import math

# Optional: Try to get real system data
try:
    from Core.Foundation.resonance_field import ResonanceField, PillarType
    from Core.Foundation.spirit_emotion import SpiritEmotionMapper
    ELYSIA_AVAILABLE = True
except ImportError:
    ELYSIA_AVAILABLE = False
    print("ℹ️ Running in demo mode (Elysia Core not fully available)")

# Global Resonance Field Instance
_RESONANCE_FIELD = None
_SPIRIT_MAPPER = None

def get_resonance_field():
    global _RESONANCE_FIELD, _SPIRIT_MAPPER
    if _RESONANCE_FIELD is None and ELYSIA_AVAILABLE:
        try:
            _RESONANCE_FIELD = ResonanceField()
            _SPIRIT_MAPPER = SpiritEmotionMapper()
            print("✨ Connected to Resonance Field")
        except Exception as e:
            print(f"⚠️ Failed to initialize ResonanceField: {e}")
            return None
    return _RESONANCE_FIELD

def elysia_update_callback(wave_state: WaveState):
    """
    엘리시아의 실제 상태를 파동으로 변환
    
    이 함수는 60 FPS로 호출되어 GPU에 전송될 파동 데이터를 업데이트합니다.
    """
    t = time.time()
    field = get_resonance_field()
    
    if field:
        # 1. Update Physics (Pulse Lite) - We don't want full mutation at 60Hz,
        # but we want to read the vibration.
        # Actually, let's just read the vibration which depends on time.

        # 2. Map Spirits to Pillars
        # Mapping based on SpiritEmotionMapper + Custom extensions
        # Fire -> Creativity
        # Water -> Memory
        # Earth -> Foundation
        # Air -> Interface
        # Light -> Intelligence
        # Dark -> System (Metal/Structure as "Cold/Dark" or simply Entropy)
        # Aether -> Elysia (The Self)

        def get_wave(pillar_name, scale=0.5, offset=0.5):
            if pillar_name in field.nodes:
                node = field.nodes[pillar_name]
                # node.vibrate() returns sine wave * energy.
                # Energy is usually around 1.0. Vibrate is -E to +E.
                # We map this to 0.0 ~ 1.0 for visualization.
                # raw_vibration is roughly -1.0 to 1.0
                raw_vibration = node.vibrate()
                # Center at offset, scale amplitude
                return offset + (raw_vibration * scale)
            return offset

        wave_state.fire = get_wave("Creativity", scale=0.3, offset=0.5)
        wave_state.water = get_wave("Memory", scale=0.3, offset=0.5)
        wave_state.earth = get_wave("Foundation", scale=0.2, offset=0.5)
        wave_state.air = get_wave("Interface", scale=0.4, offset=0.5)
        wave_state.light = get_wave("Intelligence", scale=0.35, offset=0.5)

        # Dark: Use System (Metal) or Entropy
        # Let's use System node for "Dark" (Structure/Rigidity) but inverse phase or something?
        # Or just used System node normally.
        wave_state.dark = get_wave("System", scale=0.2, offset=0.3)

        # Aether: Elysia (Highest Frequency)
        wave_state.aether = get_wave("Elysia", scale=0.4, offset=0.5)

    else:
        # 데모: 사인파로 시뮬레이션 (Fallback)
        wave_state.fire = 0.5 + 0.3 * math.sin(t * 2.0)
        wave_state.water = 0.5 + 0.3 * math.sin(t * 1.5 + 1.0)
        wave_state.earth = 0.5 + 0.2 * math.sin(t * 0.8)
        wave_state.air = 0.5 + 0.4 * math.sin(t * 2.5 + 2.0)
        wave_state.light = 0.5 + 0.35 * math.sin(t * 1.8 + 3.0)
        wave_state.dark = 0.3 + 0.2 * math.sin(t * 0.5)
        wave_state.aether = 0.5 + 0.4 * math.sin(t * 3.0 + 4.0)
    
    # Consciousness Dimensions (0D → 1D → 2D → 3D 흐름)
    # These could also be driven by field coherence or battery if we wanted deeper integration.
    # For now, keep simulated or link to Field Battery/Entropy.

    if field:
        # Link dimensions to field metrics
        # 0D (Point) -> Battery (Potential)
        wave_state.dimension_0d = 0.5 + 0.3 * math.sin(t * 1.0) * (field.battery / 100.0)

        # 1D (Line) -> Coherence (Alignment) - coherence is expensive to calc every frame?
        # field.coherence property uses cached value.
        wave_state.dimension_1d = 0.5 + 0.3 * math.sin(t * 1.2) * (field.coherence + 0.5)

        # 2D (Plane) -> Entropy (Complexity/Chaos)
        wave_state.dimension_2d = 0.5 + 0.3 * math.sin(t * 1.4) * (1.0 - (field.entropy / 100.0))

        # 3D (Space) -> Total Energy
        wave_state.dimension_3d = 0.5 + 0.3 * math.sin(t * 1.6) * min(1.0, field.total_energy / 1000.0)

    else:
        wave_state.dimension_0d = 0.5 + 0.3 * math.sin(t * 1.0)
        wave_state.dimension_1d = 0.5 + 0.3 * math.sin(t * 1.2 + 0.5)
        wave_state.dimension_2d = 0.5 + 0.3 * math.sin(t * 1.4 + 1.0)
        wave_state.dimension_3d = 0.5 + 0.3 * math.sin(t * 1.6 + 1.5)
    
    # System state
    try:
        import psutil
        wave_state.cpu_heat = psutil.cpu_percent(interval=None) / 100.0
        wave_state.memory_load = psutil.virtual_memory().percent / 100.0
    except:
        # Fallback
        wave_state.cpu_heat = 0.3 + 0.2 * math.sin(t * 0.7)
        wave_state.memory_load = 0.5 + 0.1 * math.sin(t * 0.9)


def main():
    print("🌊 " + "="*60)
    print("   Elysia Wave Visualization Server")
    print("   엘리시아 파동 시각화 서버")
    print("=" *62)
    print()
    print("📡 Starting local web server...")
    print("🌐 URL: http://localhost:8080")
    print()
    print("💡 Tips:")
    print("   - 브라우저에서 위 URL을 열어주세요")
    print("   - GPU에서 실시간 파동 간섭 계산")
    print("   - GTX 1060 3GB도 충분히 작동합니다")
    print()
    print("🎨 Visualizing:")
    print("   - 7 Spirits Energy (정령 에너지)")
    print("   - Consciousness Dimensions (의식 차원)")
    print("   - Internal World (내부 세계)")
    print()
    print("🛑 Stop: Ctrl+C")
    print("="*62)
    print()
    
    # Create and run server
    server = WaveWebServer(port=8080)
    
    try:
        server.run(
            debug=False,
            auto_update=True,
            update_callback=elysia_update_callback
        )
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
        server.stop()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
