"""
Demo: Elysia 4-Stage Interoperability Conversion Bridge
=======================================================
Demonstrates the integration between classical Von Neumann discrete computing
and continuous Complex Phasor & Rotor Field architectures.

4-Stage Pipeline Flow:
1. Stage 1: Input Encoding (Discrete bits/bytes/text/pixels -> Phasor Field Z = A e^(i phi))
2. Stage 2: Rotor Field Core Middleware (Complex phase shifts, interference, MacroRotor promotion)
3. Stage 3: In-situ Phase Error Filtering (Phase matching & destructive noise filtering)
4. Stage 4: Wave Collapse Decoding (Measurement & discrete projection)
"""

import numpy as np
from elysia_core.interoperability_bridge import InteroperabilityBridgePipeline, DiscretePhasorEncoder, WaveCollapseDecoder


def main():
    print("================================================================================")
    print("   ELYSIA 4-STAGE INTEROPERABILITY CONVERSION BRIDGE DEMO")
    print("================================================================================\n")

    pipeline = InteroperabilityBridgePipeline(chunking_threshold=0.5)

    # -------------------------------------------------------------------------
    # PART 1: Text & Byte Sequence Conversion Bridge
    # -------------------------------------------------------------------------
    print("1. [STAGE 1 -> 4] Text & Byte Stream Conversion Pipeline")
    print("--------------------------------------------------------------------------------")
    original_text = "지능공학(Intelligence Engineering): 파동과 위상 로터 아키텍처"
    raw_bytes = original_text.encode("utf-8")

    print(f"   [Input] Raw Bytes Count: {len(raw_bytes)} bytes")
    print(f"   [Input] Sample String   : '{original_text[:35]}...'")

    # Run through full 4-stage pipeline with noise injection
    rotation_shift = np.pi / 3  # 60 degrees phase shift
    noise_level = 0.05

    decoded_bytes, metrics = pipeline.process_bytes_pipeline(
        raw_bytes, rotation_shift=rotation_shift, noise_level=noise_level
    )

    decoded_text = decoded_bytes.decode("utf-8", errors="replace")

    print("\n   [Pipeline Execution Diagnostics]")
    print(f"   - Promoted MacroRotors : {metrics['promoted_macro_rotors']}")
    print(f"   - Distortion Level     : {metrics['filter_diagnostics']['global_distortion_level']:.6f} rad")
    print(f"   - Pre-Filter Coherence : {metrics['filter_diagnostics']['pre_coherence']:.4f}")
    print(f"   - Post-Filter Coherence: {metrics['filter_diagnostics']['post_coherence']:.4f}")
    print(f"   - Coherence Gain       : {metrics['filter_diagnostics']['coherence_gain']:+.4f}")
    print(f"   [Output] Recovered Text: '{decoded_text[:35]}...'")
    print(f"   - Exact Match Status   : {decoded_text == original_text}")

    # -------------------------------------------------------------------------
    # PART 2: 2D Raster Image & Geometry Field Conversion Bridge
    # -------------------------------------------------------------------------
    print("\n\n2. [STAGE 1 -> 4] 2D Raster Pixel Matrix Conversion Pipeline")
    print("--------------------------------------------------------------------------------")

    # Generate a sample 8x8 pixel matrix with a geometric gradient pattern
    y, x = np.ogrid[:8, :8]
    raster_image = np.clip((x * 30 + y * 25), 0, 255).astype(np.uint8)

    print("   [Input] Original 8x8 Pixel Grid:")
    print(raster_image)

    # Process through pipeline
    decoded_image, img_metrics = pipeline.process_image_pipeline(
        raster_image, rotation_shift=0.0, noise_level=0.02
    )

    print("\n   [Pipeline Diagnostics]")
    print(f"   - Distortion Level     : {img_metrics['filter_diagnostics']['global_distortion_level']:.6f} rad")
    print(f"   - Post-Filter Coherence: {img_metrics['filter_diagnostics']['post_coherence']:.4f}")

    print("\n   [Output] Decoded Pixel Grid After Wave Collapse:")
    print(decoded_image)

    max_pixel_diff = np.max(np.abs(decoded_image.astype(int) - raster_image.astype(int)))
    print(f"   - Max Pixel Variance   : {max_pixel_diff} / 255")

    print("\n================================================================================")
    print("   4-STAGE INTEROPERABILITY BRIDGE DEMO COMPLETED SUCCESSFULLY!")
    print("================================================================================")


if __name__ == "__main__":
    main()
