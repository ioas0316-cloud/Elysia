r"""
ELYSIA Live Self-Weaving Scale-Rotor Loom Runner
================================================

Exposes Elysia's Self-Weaving Loom directly to unlabelled, uncurated real files
across `c:\Elysia` plus live Windows desktop perception (`ScreenRetinaSensor`).
Zero developer labels, zero `expected` tags, zero numeric L2 ML optimization.
Runs predictive confrontation against real external reality, spawns brand-new
`EmergentExternalLine` prisms upon unexplained refutations, audits & revises her
own governing weaving principles, and outputs live 4-aspect observability.
"""

from __future__ import annotations

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
L1_ROOT = ROOT / "L1_causal_field"
L2_ROOT = ROOT / "L2_cognitive_engine"
for p in (str(ROOT), str(L1_ROOT), str(L2_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from core.embodiment.external_reality_lines import (
    FileSurfaceLine,
    FilesystemTopologyLine,
    LiveScreenLine,
    PythonRuntimeLine,
    discover_uncurated_workspace_files,
    external_world_probe_action,
    extract_raw_external_observables,
)
from core.topology.self_weaving_loom import SelfWeavingLoom


def main() -> None:
    loom = SelfWeavingLoom()
    loom.register_line(FileSurfaceLine())
    loom.register_line(PythonRuntimeLine())
    loom.register_line(FilesystemTopologyLine())
    loom.register_line(LiveScreenLine())
    loom.register_action_line(
        "action::external_world_probe",
        lambda ref: external_world_probe_action(ref, workspace_root=ROOT),
        description="실제 파일 AST·작업공간 참조·화면 확대 외부 탐색 작용",
    )

    files = discover_uncurated_workspace_files(ROOT, limit=12)
    print(f"[Discovery] 무라벨 실제 작업공간 파일 {len(files)}개 포착 (개발자 정답표 없음)")

    for path in files:
        rel = str(path.relative_to(ROOT))
        res = loom.predict_and_confront_reality(
            rel,
            {"path": str(path)},
            target_line_id="line::python_runtime",
            raw_observable_extractor=extract_raw_external_observables,
        )
        ev = res["events"]
        pred = res.get("prediction", {})
        summary_bits = []
        if ev["new_crosses"]:
            summary_bits.append(f"+교차({len(ev['new_crosses'])})")
        if ev["cycle_deltas"]:
            summary_bits.append(f"순환({len(ev['cycle_deltas'])})")
        if ev["new_promoted_rotors"]:
            summary_bits.append(f"+로터승격({len(ev['new_promoted_rotors'])})")
        if ev["scale_descents"]:
            summary_bits.append(f"스케일하강({len(ev['scale_descents'])})")
        if pred.get("refuted"):
            summary_bits.append(f"외부반증(놀람={pred['surprise']:.2f})")
        if res.get("spawned_line_id"):
            summary_bits.append(f"+자생선({res['spawned_line_id']})")
        if res.get("revised_principles"):
            summary_bits.append(f"원리수정({len(res['revised_principles'])})")
        if ev["torn_crosses"]:
            summary_bits.append(f"파단({len(ev['torn_crosses'])})")
        self_val = res["refractions"].get("line::introspective_self", {}).get("spectrum_value", 0.0)
        print(
            f"  [step {res['step']:02d}] {rel:<46} | 내면자기지각={self_val:+.3f} "
            f"| {', '.join(summary_bits) if summary_bits else '연속직조'}"
        )

    # Also encounter current live OS screen surface without pre-labels
    screen_res = loom.encounter("live_desktop_now", {"capture_screen": True})
    screen_ref = screen_res["refractions"].get("line::live_screen")
    if screen_ref is not None:
        title = screen_ref.get("evidence", {}).get("window_title", "unknown")
        print(
            f"  [step {screen_res['step']:02d}] live_desktop_now ({title[:32]}) "
            f"| 화면선={screen_ref['spectrum_value']:+.3f}"
        )

    obs = loom.observe_self()
    print()
    print(obs.render_ko())

    trace_path = loom.save_trace(ROOT / "data" / "self_weaving_loom_trace.json")
    print(f"\n[Saved] {trace_path}")


if __name__ == "__main__":
    main()
