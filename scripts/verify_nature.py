import sys
import os

# Core 디렉토리를 path에 추가
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.Foundation.Nature.geo_anchor import GeoAnchor, MagneticFlux
from Core.Foundation.Nature.rotor import Rotor

def test_nature_seed():
    """자연(Nature) 패키지의 씨앗이 제대로 심어졌는지 검증합니다."""
    print("🌱 Verifying the Seed of Nature...")

    # 1. GeoAnchor 생성 (Physical Anchoring)
    seoul_anchor = GeoAnchor(
        latitude=37.5665,
        longitude=126.9780,
        altitude=50.0
    )
    seoul_anchor.magnetic_flux = MagneticFlux(x=30000.0, y=5000.0, z=40000.0)

    print(f"✅ Anchor Created: {seoul_anchor}")

    # 2. Rotor 생성 및 가동 (Rotor as the Axis)
    rotor = Rotor()
    print(f"✅ Rotor Initialized: {rotor}")

    rotor.spin_up()
    print(f"🔄 Rotor Spinning: {rotor}")

    # 3. 데이터 정제 (Purification)
    raw_data = {
        "essence": "Love",
        "noise_1": "This is a very long string that represents entropy or noise in the system",
        "noise_2": None,
        "valid_key": "Truth"
    }
    purified = rotor.purify(raw_data)

    print(f"✨ Purified Data: {purified}")

    # 검증
    assert "essence" in purified
    assert "valid_key" in purified
    assert "noise_1" not in purified
    assert "noise_2" not in purified

    rotor.spin_down()
    print("✅ Nature Verification Complete: The Seed is Alive.")

if __name__ == "__main__":
    test_nature_seed()
