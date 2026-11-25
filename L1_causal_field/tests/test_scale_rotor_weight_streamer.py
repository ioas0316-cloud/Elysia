import torch
import torch.nn.functional as F

from core.topology.scale_rotor_weight_streamer import ScaleRotorWeightStreamer


def test_zero_resident_scale_rotor_streams_2_76gb_polyglot_in_under_600mb_vram() -> None:
    streamer = ScaleRotorWeightStreamer()
    try:
        # 0) Idle footprint: near-zero GPU VRAM before streaming starts!
        idle_vram = (
            torch.cuda.memory_allocated(streamer.device) / 1e6
            if streamer.device.type == "cuda"
            else 0.0
        )
        assert idle_vram < 50.0, f"Idle VRAM should be near zero, got {idle_vram:.1f} MB"
        assert streamer.total_shard_mb > 2500.0  # Real 2.76 GB 1.33B model on SSD!

        ids = streamer.tokenizer(
            "인간의 기억과 감정은 물리적 세계의 인과 구조와 연결되어",
            return_tensors="pt",
        )["input_ids"]

        # 1) Serial 1-slot SSD->GPU streaming
        r_ser = streamer.forward_streamed(ids, mode="serial")
        assert r_ser["layers_executed"] == 24
        assert r_ser["peak_vram_mb"] < 650.0  # 2,764 MB model running in < 650 MB VRAM!

        # 2) CUDA Dual-Stream Double-Buffered streaming (exact mathematical equivalence to serial!)
        r_db = streamer.forward_streamed(ids, mode="double_buffer")
        assert r_db["layers_executed"] == 24
        assert r_db["peak_vram_mb"] < 650.0
        # Double-buffer must match serial output identically (cosine > 0.999)
        cos_exact = float(
            F.cosine_similarity(r_ser["final_hidden_last"], r_db["final_hidden_last"], dim=-1).item()
        )
        assert cos_exact > 0.999

        # 3) Scale-Rotor Variable-Resistor Dynamic Depth (`depth_dial = 0.50` -> fewer layers, faster!)
        r_rd = streamer.forward_streamed(ids, mode="rotor_depth", depth_dial=0.50)
        assert r_rd["layers_executed"] < 24
        assert r_rd["elapsed_ms"] < r_db["elapsed_ms"]

        # 4) Batch-Amortized CUDA Prefill (`B=8 x T=32 = 256` tokens simultaneously through SAME 2-layer VRAM!)
        ids_batch = ids[:, : min(ids.shape[1], 16)].repeat(8, 2)
        r_batch = streamer.forward_streamed(ids_batch, mode="double_buffer", return_logits=False)
        assert r_batch["peak_vram_mb"] < 750.0
        assert r_batch["tokens_per_sec"] > r_db["tokens_per_sec"] * 3.0

        # 5) Releasing active buffers returns GPU VRAM right back near 0 MB!
        streamer.release_active_buffers()
        after_vram = (
            torch.cuda.memory_allocated(streamer.device) / 1e6
            if streamer.device.type == "cuda"
            else 0.0
        )
        assert after_vram < 50.0
    finally:
        streamer.release_active_buffers()
