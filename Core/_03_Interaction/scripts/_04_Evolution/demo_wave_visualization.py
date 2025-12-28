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

from Core._03_Interaction._01_Interface.Interface.wave_web_server import WaveWebServer, WaveState
import time
import math

# Optional: Try to get real system data
try:
    from Core._01_Foundation._05_Governance.Foundation.resonance_field import ResonanceField
    from Core._01_Foundation._05_Governance.Foundation.digital_ecosystem import DigitalEcosystem
    ELYSIA_AVAILABLE = True
except ImportError:
    ELYSIA_AVAILABLE = False
    print("ℹ️ Running in demo mode (Elysia Core not fully available)")


def elysia_update_callback(wave_state: WaveState):
    """
    엘리시아의 실제 상태를 파동으로 변환
    
    이 함수는 60 FPS로 호출되어 GPU에 전송될 파동 데이터를 업데이트합니다.
    """
    t = time.time()
    
    if ELYSIA_AVAILABLE:
        # TODO: 실제 ResonanceField에서 에너지 가져오기
        # resonance = ResonanceField()
        # wave_state.fire = resonance.get_spirit_energy("Fire")
        # ...
        pass
    
    # 데모: 사인파로 시뮬레이션
    wave_state.fire = 0.5 + 0.3 * math.sin(t * 2.0)
    wave_state.water = 0.5 + 0.3 * math.sin(t * 1.5 + 1.0)
    wave_state.earth = 0.5 + 0.2 * math.sin(t * 0.8)
    wave_state.air = 0.5 + 0.4 * math.sin(t * 2.5 + 2.0)
    wave_state.light = 0.5 + 0.35 * math.sin(t * 1.8 + 3.0)
    wave_state.dark = 0.3 + 0.2 * math.sin(t * 0.5)
    wave_state.aether = 0.5 + 0.4 * math.sin(t * 3.0 + 4.0)
    
    # Consciousness Dimensions (0D → 1D → 2D → 3D 흐름)
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
