"""
Epistemic Deficit Harvester & Causal Language Bridge Empirical Demo
===================================================================

Demonstrates:
1. Self-awareness of epistemic deficit ("what principle/tool is missing from
   my palette?"), read-only AST exploration across `L1_causal_field/core`,
   and real dynamic internalization of `HangulTopologicalEngine`.
2. Grounding Korean language into Hunminjeongeum articulatory wave-rotors +
   Chinese-Room disconnection tracing -> driving ElysiaSelfObservingOrchestra.
"""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import elysia_loader

elysia_loader.install()

from core.orchestration.elysia_self_observing_orchestra import (
    ElysiaSelfObservingOrchestra,
)
from core.orchestration.epistemic_harvester_and_language_bridge import (
    CausalLanguagePaletteBridge,
    EpistemicToolHarvester,
)


def run_demo() -> None:
    print("=" * 94)
    print("     PHASE 6: EPISTEMIC DEFICIT HARVESTER & CAUSAL LANGUAGE-PALETTE BRIDGE DEMO")
    print("=" * 94)

    bridge = CausalLanguagePaletteBridge(harvester=EpistemicToolHarvester())

    with tempfile.TemporaryDirectory(prefix="elysia_lang_") as tmp_dir:
        slab = Path(tmp_dir) / "lang_cinema.slab"
        with ElysiaSelfObservingOrchestra(slab_path=slab, slice_duration_s=0.6) as director:
            utterances = [
                "새벽의 부드럽고 평온한 빛과 화음",
                "황혼의 강렬한 한글 조음 폭풍과 프리즘 대비",
                "한글 조음 파동으로 정오의 맑은 빛과 소리",
            ]

            for idx, text in enumerate(utterances, start=1):
                res = bridge.conduct_from_language(
                    text, director=director, auto_harvest=True, manifest=False
                )
                assert res.directing_result is not None
                drm = res.directing_result.present_slice.performances[
                    "narrative.drama_director"
                ].causal_state
                disc = res.tethering_report["experiential_disconnection"]

                print(f"\n[Utterance #{idx}] \"{text}\"")
                print(f"  Deficit Check   : has_deficit={res.deficit_diagnosis.has_deficit}, missing={res.deficit_diagnosis.missing_concepts}")
                if res.harvested_pigment_id:
                    print(f"  Tool Harvested  : {res.harvested_pigment_id}")
                print(
                    f"  Hangul Physics  : friction={res.articulatory_metrics['mean_friction']:.3f}, "
                    f"tension={res.articulatory_metrics['mean_tension']:.3f}, "
                    f"vowel_phase={res.articulatory_metrics['mean_vowel_phase']:.3f} rad, "
                    f"knots={int(res.articulatory_metrics['high_friction_knots'])}"
                )
                print(
                    f"  Orchestra Act   : {drm['act_title']} (t={res.inferred_time:.1f}, "
                    f"scale={res.inferred_balance.scale_tilt:+.2f}, "
                    f"contrast={res.inferred_balance.contrast_tilt:+.2f}, "
                    f"ChineseRoom_disconnection={disc:.3f})"
                )
                print(f"  Grounded Reply  : {res.grounded_response_ko}")

            print("\n" + "-" * 94)
            print(f"Harvest Log Entries: {len(bridge.harvester.harvest_log)}")
            for entry in bridge.harvester.harvest_log:
                print(f"  - {entry['pigment_id']} <- {entry['source_file']} ({entry['class_name']})")
            print("=" * 94)


if __name__ == "__main__":
    run_demo()
