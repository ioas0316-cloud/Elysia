import os
import time
import tracemalloc
import zlib
import numpy as np
from PIL import Image
from core.topology.binary_container_stripper import PNGContainerStripper

def create_noisy_png(target_bytes: int, filename: str):
    """
    고엔트로피(노이즈/실제 사진 특성) 데이터를 포함한 PNG 파일 생성
    """
    print(f"Generating noisy PNG target size ~{target_bytes / (1024*1024):.2f} MB -> {filename}")

    # 목표 크기에 맞게 이미지 가로/세로 해상도 계산
    # RGB 3바이트/픽셀, 노이즈 축소율 감안
    approx_pixels = target_bytes / 3.0
    side = int(np.sqrt(approx_pixels))
    side = max(16, side)

    # 노이즈 + 구조화된 패턴 합성 (실제 노이즈 사진 특성)
    np.random.seed(42)
    noise_data = np.random.randint(0, 256, (side, side, 3), dtype=np.uint8)

    img = Image.fromarray(noise_data, 'RGB')

    # compression_level을 조정하여 정확한 크기에 맞춤
    img.save(filename, format="PNG", compress_level=1)

    actual_size = os.path.getsize(filename)
    print(f"Generated {filename}: actual size = {actual_size} bytes ({actual_size / (1024*1024):.2f} MB)")
    return actual_size

def benchmark():
    os.makedirs("test_pngs", exist_ok=True)

    files_to_test = [
        ("1KB", 1024, "test_pngs/test_1kb.png"),
        ("1MB", 1024 * 1024, "test_pngs/test_1mb.png"),
        ("100MB", 100 * 1024 * 1024, "test_pngs/test_100mb.png"),
    ]

    stripper = PNGContainerStripper()
    results = []

    for name, target_size, filepath in files_to_test:
        create_noisy_png(target_size, filepath)

        with open(filepath, "rb") as f:
            raw_bytes = f.read()

        actual_size_mb = len(raw_bytes) / (1024 * 1024)

        # Tracemalloc & Timer Start
        tracemalloc.start()
        start_time = time.perf_counter()

        parsed = stripper.parse(raw_bytes)
        reassembled = stripper.reassemble(parsed)

        elapsed_time = time.perf_counter() - start_time
        current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        peak_mem_mb = peak_mem / (1024 * 1024)
        is_match = (reassembled == raw_bytes)

        print(f"[{name}] Size: {actual_size_mb:.2f} MB | Match: {is_match} | Time: {elapsed_time:.4f}s | Peak Mem: {peak_mem_mb:.2f} MB")

        results.append({
            "label": name,
            "actual_size_bytes": len(raw_bytes),
            "actual_size_mb": actual_size_mb,
            "time_sec": elapsed_time,
            "peak_mem_mb": peak_mem_mb,
            "match": is_match,
            "num_chunks": len(parsed["chunks"])
        })

    print("\n=== BENCHMARK RESULTS SUMMARY ===")
    for r in results:
        print(f"Label: {r['label']} ({r['actual_size_mb']:.2f} MB) | Parse+Reassemble Time: {r['time_sec']:.4f}s | Peak Memory: {r['peak_mem_mb']:.2f} MB | Match: {r['match']} | Chunks: {r['num_chunks']}")

if __name__ == "__main__":
    benchmark()
