from __future__ import annotations

import json
from pathlib import Path

from core.sensory.real_world_corpus_comprehension_engine import (
    DEFAULT_CORPUS_DASHBOARD_PATH,
    DEFAULT_CORPUS_ROOT,
    DEFAULT_CORPUS_STATE_PATH,
    RealWorldCorpusComprehensionEngine,
    extract_blind_multi_scale_signature,
    render_real_world_corpus_dashboard,
)


def test_real_world_workspace_comprehension_benchmark() -> None:
    """
    Zero-toy empirical benchmark directly on REAL files in `c:\\Elysia`:
      1. Ingests >= 180 real files across `.py`, `.cpp/.cu/.h/.hlsl`, `.md`, `.json`, `.png`.
      2. Achieves strong held-out blind modality classification (>= 85%) from raw bytes ONLY.
      3. Achieves strong Korean+English conceptual retrieval (`MRR >= 0.70`) & discovers genuine
         cross-format twins (`.md <-> .py`, `.py <-> .cu/.h`).
      4. Cross-infers masked MDL compressibility strictly from other structural dims (`r >= 0.65`).
      5. Rejects injected random binary, echo bloat, and syntax corruption (`>= 1.5x` distance).
      6. Saves telemetry JSON & renders crisp 4-panel dashboard PNG.
    """
    engine = RealWorldCorpusComprehensionEngine(
        corpus_root=DEFAULT_CORPUS_ROOT,
        state_path=DEFAULT_CORPUS_STATE_PATH,
    )
    report = engine.run_real_world_benchmark(max_per_modality=90)

    print("\n=== [ELYSIA Real-World Workspace Comprehension Log] ===")
    print(report.summary_ko)

    # Verify filename-free blind signature extractor truly uses bytes only (identical bytes -> identical vec)
    sample_py = (DEFAULT_CORPUS_ROOT / r"L1_causal_field\core\physics\causal_field.py").read_bytes()
    v1, t1 = extract_blind_multi_scale_signature(sample_py)
    v2, t2 = extract_blind_multi_scale_signature(sample_py)
    assert (v1 == v2).all() and t1 == t2

    # 1. Real corpus ingestion coverage
    assert report.total_files_ingested >= 180
    assert report.test_count >= 50
    for mod, cnt in report.modality_counts.items():
        assert cnt >= 15, f"Too few real files for modality {mod}: {cnt}"

    # 2. Blind held-out modality classification accuracy
    assert report.blind_modality_accuracy >= 0.85

    # 3. Korean+English conceptual retrieval & cross-format twins
    assert report.retrieval_mrr >= 0.70
    assert report.retrieval_top3_rate >= 0.80
    assert len(report.cross_format_twins) == 2
    for tw in report.cross_format_twins:
        assert tw.resonance_score > 0.15
        assert len(tw.shared_concepts) >= 2

    # 4. Blind MDL compressibility cross-inference
    assert report.blind_mdl_pearson_r >= 0.65
    assert report.blind_mdl_mae <= 0.12

    # 5. Real anomaly separation
    for anom in report.anomaly_results:
        assert anom.flagged_anomalous is True
        assert anom.ratio_vs_normal >= 1.50

    # 6. CausalField & JSON & Visual Dashboard checks
    assert report.causal_field_voxels == report.total_files_ingested
    assert report.causal_field_beams >= report.total_files_ingested
    state_file = Path(report.state_json_path)
    assert state_file.exists() and state_file.stat().st_size > 2_000
    saved = json.loads(state_file.read_text(encoding="utf-8"))
    assert saved["total_files_ingested"] == report.total_files_ingested

    dash_path = render_real_world_corpus_dashboard(
        engine=engine,
        report=report,
        output_path=DEFAULT_CORPUS_DASHBOARD_PATH,
    )
    assert dash_path.exists() and dash_path.stat().st_size > 20_000
