from pathlib import Path
import time

import numpy as np
from PIL import Image
import torch

from core.topology.interlocked_scale_rotor_engine import (
    AsyncMultimodalWorldFeeder,
    InterlockedScaleRotorEngine,
    WorldKnowledgeMeta,
)


def test_interlocked_5_scale_gear_train_speed_plasticity_and_persistence(tmp_path: Path) -> None:
    engine = InterlockedScaleRotorEngine(tmp_path / "rotor_fast", use_pretrained=False)

    # Seed 500 synthetic multimodal items into GPU/CPU K_bank to test real tensor-bank resonance speed
    g = torch.Generator(device="cpu").manual_seed(123)
    vecs = torch.randn(500, engine.dim, generator=g)
    metas = [
        WorldKnowledgeMeta(
            item_id=f"item::{i}",
            modality=("image" if i % 3 == 0 else "markdown"),
            source_path=f"/mock/{i}",
            title=f"concept_{i}",
            preview=f"preview_{i}",
            timestamp=0.0,
        )
        for i in range(500)
    ]
    assert engine.add_knowledge_batch(vecs, metas) == 500

    wave = engine.transducer.encode_texts(["gravity metabolism balance continuous field"])[0]
    first = engine.step_interlocked_frame(wave, plastic=True, return_resonance_top_k=3)

    t0 = time.perf_counter()
    n_steps = 120
    for _ in range(n_steps):
        engine.step_interlocked_frame(wave, plastic=True)
    elapsed = time.perf_counter() - t0
    hz = n_steps / max(1e-6, elapsed)

    last = engine.step_interlocked_frame(wave, plastic=True, return_resonance_top_k=3)
    tel = engine.telemetry()

    assert hz >= 100.0, f"Expected high-speed interlocked gear-train >=100Hz, got {hz:.1f}Hz"
    assert last["s0_pred_error"] < first["s0_pred_error"]
    assert tel["drift_W_down_norm"] > 0.0
    assert tel["drift_R_norm"] > 0.0
    assert not torch.isnan(engine.S).any()

    # Compact binary save & reload preserves all learned gears, W, R, and 500-item K_bank
    engine.save()
    reloaded = InterlockedScaleRotorEngine(tmp_path / "rotor_fast", use_pretrained=False)
    assert reloaded.K_bank.shape[0] == 500
    assert reloaded.frame_count == engine.frame_count
    assert torch.allclose(reloaded.S, engine.S, atol=1e-4)
    assert torch.allclose(reloaded.R, engine.R, atol=1e-4)


def test_real_pretrained_gpu_clip_and_minilm_bulk_multimodal_ingestion(tmp_path: Path) -> None:
    engine = InterlockedScaleRotorEngine(tmp_path / "rotor_pretrained", use_pretrained=True)
    assert engine.transducer.pretrained_loaded is True, "Local CLIP + MiniLM cache should load offline"

    feeder = AsyncMultimodalWorldFeeder(engine)
    bulk = feeder.ingest_workspace_bulk(max_docs=40, max_code=40, max_images=12)
    assert bulk["total_added"] >= 50
    tel = engine.telemetry()
    assert tel["modality_counts"].get("markdown", 0) >= 10
    assert tel["modality_counts"].get("code", 0) >= 10
    assert tel["modality_counts"].get("image", 0) >= 5

    # Korean semantic query resonates directly across ingested real documents
    q_ko = engine.query_multimodal(text="감정과 불협화음 그리고 기억의 존재론", settle_frames=10, top_k=5)
    assert len(q_ko["direct_matches"]) == 5
    assert q_ko["direct_matches"][0]["similarity"] > 0.25

    # Real RGB image query enters the EXACT same 896-D multimodal gear-train without errors
    img = Image.fromarray(np.full((128, 128, 3), 180, dtype=np.uint8))
    q_img = engine.query_multimodal(image=img, settle_frames=10, top_k=3)
    assert len(q_img["direct_matches"]) == 3
    assert engine.frame_count >= 20
