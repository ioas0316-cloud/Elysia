from pathlib import Path

import numpy as np

from core.topology.gigascale_media_rotor import GigascaleMediaRotorPipeline
from core.topology.hardware_scale_rotor_memory import HardwareScaleRotorMemory
from core.topology.korean_causal_linguistics import attach_particle

REAL_4K_PHOTO = Path(r"c:\Elysia\demos\real_world_assets\genuine_optical_4k_yosemite.jpg")
REAL_105MB_BIN = Path(r"c:\Archive\03_Traces_and_Deprecated\_Data_Archive\data_deprecated\yggdrasil_torus.bin")


def test_parenthetical_korean_particle_uses_head_noun() -> None:
    assert attach_particle("1501프레임(예측오차 0.8712)", "을/를") == "1501프레임(예측오차 0.8712)을"
    assert attach_particle("SSD(거시 영구 저장)", "은/는") == "SSD(거시 영구 저장)는"
    assert attach_particle("VRAM(미시 핫 뱅크)", "이/가") == "VRAM(미시 핫 뱅크)이"


def test_gigascale_12mp_and_8k_33mp_tiled_cuda_rotor_bounded_vram() -> None:
    assert REAL_4K_PHOTO.is_file(), f"Missing real 4K photo: {REAL_4K_PHOTO}"
    pipe = GigascaleMediaRotorPipeline()

    # 1) 12-Megapixel (4000x3000) real optical image tiled scan on GTX 1060 CUDA
    res_12mp = pipe.process_ultra_res_image_tiled(
        REAL_4K_PHOTO,
        upscale_to=(4000, 3000),
        tile_size=512,
        tile_batch=8,
    )
    assert res_12mp["ok"] is True
    assert res_12mp["megapixels"] == 12.0
    assert res_12mp["tiles_scanned"] >= 48
    assert res_12mp["mean_edge_energy"] > 0.01
    assert res_12mp["peak_vram_delta_mb"] < 250.0  # Strictly bounded well below 3 GB GTX 1060!

    # 2) True 8K UHD (7680x4320 = 33.17 Megapixels, 398 MB float32!) tiled scan on GTX 1060 CUDA!
    res_8k = pipe.process_ultra_res_image_tiled(
        REAL_4K_PHOTO,
        upscale_to=(7680, 4320),
        tile_size=512,
        tile_batch=8,
    )
    assert res_8k["ok"] is True
    assert res_8k["megapixels"] > 33.0
    assert res_8k["raw_float32_mb"] > 375.0
    assert res_8k["tiles_scanned"] >= 120
    assert res_8k["peak_vram_delta_mb"] < 250.0  # Processed 398MB float32 8K frame in <250MB VRAM!


def test_gigascale_real_4k_optical_video_stream_and_105mb_plus_1gb_file_array_rotor(tmp_path: Path) -> None:
    mem = HardwareScaleRotorMemory(tmp_path / "giga_mem", dim=896, max_vram=64, max_ram=128)
    pipe = GigascaleMediaRotorPipeline(memory=mem)
    try:
        # 1) Build & stream a REAL 4K (3840x2160 x 16 frames = 132.7 Megapixels, ~398 MB raw RGB) optical video!
        mp4_path = tmp_path / "real_4k_yosemite_pan.mp4"
        built = pipe.build_real_4k_optical_video_from_photo(
            REAL_4K_PHOTO, mp4_path, num_frames=16, fps=24, target_wh=(3840, 2160)
        )
        assert built["ok"] is True and mp4_path.is_file()

        vid_res = pipe.stream_4k_video_rotor(mp4_path, max_frames=16, tile_size=512, tile_batch=8)
        assert vid_res["ok"] is True
        assert vid_res["resolution"] == [3840, 2160]
        assert vid_res["frames_processed"] == 16
        assert vid_res["total_megapixels_processed"] > 130.0
        assert vid_res["mean_temporal_flux"] > 0.005
        assert vid_res["peak_vram_delta_mb"] < 250.0

        # 2) Stream the REAL 104.86 MB binary file already in Archive (`yggdrasil_torus.bin`)!
        if REAL_105MB_BIN.is_file():
            r105 = pipe.stream_gigabyte_file_array_rotor(REAL_105MB_BIN, slab_mb=16, interpret_dtype="uint8")
            assert r105["ok"] is True
            assert r105["file_size_mb"] >= 100.0
            assert r105["peak_vram_delta_mb"] < 250.0
            assert r105["promoted_to_memory"] >= 1

        # 3) Stream a REAL 1.0 GB (1,024 MB = 268,435,456 float32s) SSD memmap file with a buried anomaly at +736 MB!
        gb_path = tmp_path / "real_1gb_rotor_array.bin"
        total_floats = (1024 * 1024 * 1024) // 4  # exact 1.000 GiB
        mm = np.memmap(gb_path, dtype=np.float32, mode="w+", shape=(total_floats,))
        # Fill cleanly in 16MiB slabs (`4M float32 = 16 MiB` x 64 = exact 1.000 GiB), with buried anomaly at slab #46 (+736 MiB)!
        bg = np.full((4 * 1024 * 1024,), 0.05, dtype=np.float32)
        for s_i in range(64):
            s0 = s_i * len(bg)
            mm[s0 : s0 + len(bg)] = bg
        # Bury strong high-contrast anomaly strictly inside slab #46 (736 MiB .. 752 MiB)!
        anom_start = 46 * len(bg)
        mm[anom_start : anom_start + len(bg)] = np.linspace(-8.0, 8.0, len(bg), dtype=np.float32)
        mm.flush()
        del mm

        r1gb = pipe.stream_gigabyte_file_array_rotor(gb_path, slab_mb=16, interpret_dtype="float32")
        assert r1gb["ok"] is True
        assert r1gb["file_size_mb"] >= 1024.0
        assert r1gb["slabs_streamed"] == 64
        assert r1gb["peak_vram_delta_mb"] < 250.0  # Full 1.0 GiB streamed on 3GB GTX 1060 in <250MB VRAM!
        # Must pinpoint exact buried anomaly slab #46 (+736 MB) inside the 1.0 GiB file!
        assert r1gb["top_anomaly_slabs"][0]["slab_idx"] == 46
        assert r1gb["top_anomaly_slabs"][0]["anomaly_z"] > 5.0
        assert r1gb["promoted_to_memory"] >= 1
    finally:
        mem.close()
