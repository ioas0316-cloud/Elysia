from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import elysia_loader
elysia_loader.install()

from core.sensory.real_world_corpus_comprehension_engine import (
    DEFAULT_CORPUS_DASHBOARD_PATH,
    DEFAULT_CORPUS_ROOT,
    DEFAULT_CORPUS_STATE_PATH,
    RealWorldCorpusComprehensionEngine,
    render_real_world_corpus_dashboard,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ELYSIA Real-World Multi-Modal Workspace Comprehension Benchmark & Live Query"
    )
    parser.add_argument(
        "--query",
        type=str,
        default="",
        help="Optional custom Korean or English question to query the live ingested workspace",
    )
    parser.add_argument(
        "--max-per-modality",
        type=int,
        default=90,
        help="Max strided files ingested per modality from c:\\Elysia",
    )
    args = parser.parse_args()

    engine = RealWorldCorpusComprehensionEngine(
        corpus_root=DEFAULT_CORPUS_ROOT,
        state_path=DEFAULT_CORPUS_STATE_PATH,
    )
    report = engine.run_real_world_benchmark(max_per_modality=args.max_per_modality)
    dash_path = render_real_world_corpus_dashboard(
        engine=engine,
        report=report,
        output_path=DEFAULT_CORPUS_DASHBOARD_PATH,
    )

    print("\n" + "=" * 88)
    print("ELYSIA Real-World Multi-Modal Corpus Comprehension Benchmark (c:\\Elysia)")
    print("=" * 88)
    print(report.summary_ko)
    print(f"  Modality Distribution: {report.modality_counts}")

    print("\n[Benchmark 1: Blind Held-Out Modality Recall (0% Filename Info)]")
    for mod, rec in report.per_class_recall.items():
        print(f"  - {mod:20s}: recall={rec * 100:.1f}%")

    print("\n[Benchmark 2a: Real Korean+English Conceptual Retrieval]")
    for q, hits in report.query_examples.items():
        print(f"\n  Query: \"{q}\"")
        for h in hits[:3]:
            print(f"     #{h.rank} [{h.modality:16s}] score={h.score:.3f} | {h.rel_path}")

    print("\n[Benchmark 2b: Discovered Cross-Format Isomorphism Twins]")
    for tw in report.cross_format_twins:
        print(
            f"  * {tw.left_path} ({tw.left_modality}) <==> {tw.right_path} ({tw.right_modality}) "
            f"| score={tw.resonance_score:.3f} | shared={tw.shared_concepts}"
        )

    print("\n[Benchmark 3: Blind MDL Compressibility Cross-Inference]")
    print(f"  Pearson r = {report.blind_mdl_pearson_r:.3f} | MAE = {report.blind_mdl_mae:.4f}")

    print(f"\n[Benchmark 4: Live Anomaly Rejection (Normal Held-Out Dist = {report.normal_baseline_distance:.3f})]")
    for anom in report.anomaly_results:
        print(
            f"  - {anom.label:38s}: dist={anom.mean_distance:.3f} "
            f"({anom.ratio_vs_normal:.2f}x normal) | flagged={anom.flagged_anomalous}"
        )

    print("\n[Benchmark 5: Empirical Corpus Scaling Curve]")
    for pt in report.scaling_curve:
        print(
            f"  - Train Files={pt.train_doc_count:3d} -> Blind Acc={pt.blind_modality_accuracy * 100:.1f}% | "
            f"Retrieval MRR={pt.retrieval_mrr:.3f}"
        )

    if args.query.strip():
        print(f"\n[Custom Live Query Result: \"{args.query}\"]")
        for h in engine.query_corpus(args.query, top_k=5):
            print(f"  #{h.rank} [{h.modality:16s}] score={h.score:.3f} | {h.rel_path}")
            print(f"      {h.preview}")

    print(f"\n[Saved State JSON] {report.state_json_path}")
    print(f"[Saved Dashboard]  {dash_path}")


if __name__ == "__main__":
    main()
