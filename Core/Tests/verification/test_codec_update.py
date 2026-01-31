import numpy as np
from Core.1_Body.L1_Foundation.Logic.qualia_7d_codec import codec

def test_codec():
    print("Testing Codec Update...")

    # 1. Test Encoding (New Standard)
    vec = np.array([1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0]) # A R V A R V A
    encoded = codec.encode_sequence(vec)
    print(f"Encoded: {encoded}")
    assert encoded == "ARVARVA"

    # 2. Test Decoding (New Standard)
    decoded = codec.decode_sequence("ARVARVA")
    print(f"Decoded: {decoded}")
    expected = np.array([1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0])
    expected = expected / np.linalg.norm(expected)

    assert np.allclose(decoded, expected, atol=0.01)

    # 3. Test Backward Compatibility
    legacy_encoded = "HDVHDVH"
    decoded_legacy = codec.decode_sequence(legacy_encoded)
    print(f"Decoded Legacy: {decoded_legacy}")
    assert np.allclose(decoded_legacy, expected, atol=0.01)

    print("Codec Verified.")

if __name__ == "__main__":
    test_codec()
