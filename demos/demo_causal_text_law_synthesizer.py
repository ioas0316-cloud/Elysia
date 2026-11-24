r"""
ELYSIA Phase 12 Demo: Natural-Language Causal Law Synthesizer, Dimensional Mirror,
Multi-World Constant Discovery & Honest Metaphor Back-Tracing
==================================================================================

Demonstrates Elysia reading raw mixed Korean/English prose and filtering every
sentence through Dimensional Mirror, Multi-World Embodied Verification, and Honest
Metaphor Self-Reflection:
1. Equation-free prose ("최대 방출 파장은 온도에 반비례한다") -> discovers Wien's
   constant `2,897,771.955` across [3200K, 5800K] and transfers zero-shot to 4500K!
2. Acoustic inverse-square prose ("소리 세기는 거리의 제곱에 반비례한다") ->
   discovers `P / 4*pi = 3.1831` across [5m, 10m]!
3. Explicit equation (`lambda * T = 2897771.955`) & Unsafe AST injection guard.
4. Pre-experimental Dimensional Mirror rejection ("파장 + 온도 = 500").
5. Multi-world empirical falsification ("파장도 온도에 비례해서 커진다").
6. Honest literal limits + grounded physical back-tracing on poetic metaphors
   ("그의 분노는 붉은 용광로처럼 뜨겁게 타오른다", "침묵은 무거운 돌처럼...").
7. CausalField projection with abstract-sky curiosity charge.
"""

from __future__ import annotations

from pathlib import Path
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import elysia_loader

elysia_loader.install()

from core.orchestration.causal_text_law_synthesizer import (
    CausalTextLawSynthesizer,
    sync_synthesis_to_causal_field,
)
from core.orchestration.extended_cognitive_organism import (
    ExtendedCognitiveOrganism,
    ExtendedWorldState,
    WIEN_CONSTANT_NM_K,
)
from core.physics.causal_field import CausalField


def _section(title: str) -> None:
    print("\n" + "=" * 92)
    print(f"  {title}")
    print("=" * 92)


def run_demo() -> None:
    print("=" * 92)
    print("  ELYSIA LIVING CAUSAL TEXT SYNTHESIZER (PHASE 12) — PROSE TO BODILY DISCIPLINE")
    print("=" * 92)

    synth = CausalTextLawSynthesizer()
    org = ExtendedCognitiveOrganism()
    org.graft_spectrometer()

    wien_worlds = [ExtendedWorldState(source_temp_k=3200.0), ExtendedWorldState(source_temp_k=5800.0)]

    _section("1. MIXED DOCUMENT INGESTION (EQUATION-FREE LAW, MYTH, DIMENSION TRAP, METAPHOR)")
    doc = "\n".join(
        [
            "빈의 변위 법칙에 따르면 방출 파장과 온도의 곱은 같다: `lambda * T = 2897771.955`",
            "물체의 절대온도가 높아질수록 최대 방출 파장은 온도에 반비례한다.",
            "파장 + 온도 = 500",
            "물체의 온도가 높아질수록 최대 방출 파장도 온도에 비례해서 커진다.",
            "그의 분노는 붉은 용광로처럼 뜨겁게 타오른다.",
            "침묵은 무거운 돌처럼 방 안에 내려앉았다.",
            "__import__('os').system('echo pwn') = 1",
        ]
    )
    report = synth.ingest_document_text(doc, org, wien_worlds)
    for res in report.results:
        print(f"\nText: \"{res.raw_text}\"")
        print(f"  -> Outcome={res.outcome.value} | skipped_exp={res.experiment_skipped} | "
              f"action={res.multi_world_action}")
        if res.discovered_constant is not None:
            print(f"  -> Discovered Invariant Constant = {res.discovered_constant:.6f} "
                  f"(CODATA Wien = {WIEN_CONSTANT_NM_K:.3f})")
        print(f"  -> {res.explanation}")

    _section("2. EQUATION-FREE ACOUSTIC INVERSE-SQUARE CONSTANT DISCOVERY")
    dist_worlds = [
        ExtendedWorldState(source_pos_x=5.0, source_power_w=40.0),
        ExtendedWorldState(source_pos_x=10.0, source_power_w=40.0),
    ]
    res_sound = synth.synthesize_and_verify(
        text="거리가 멀어질수록 소리 세기는 거리의 제곱에 반비례한다.",
        organism=org,
        calibration_worlds=dist_worlds,
        use_spectrometer=False,
    )
    print(f"Text: \"{res_sound.raw_text}\"")
    print(f"  -> {res_sound.explanation}")
    print(f"  -> Discovered Constant = {res_sound.discovered_constant:.6f} (40/(4*pi) = {40.0 / (4.0 * 3.141592653589793):.6f})")

    _section("3. ZERO-SHOT 4500K TRANSFER WITHOUT SPECTROMETER & CAUSAL FIELD SYNC")
    org.detach_tool("spectrometer")
    unseen_4500 = ExtendedWorldState(source_temp_k=4500.0)
    perc = org.perceive_extended(unseen_4500, active_organs=("ear", "hand"))
    print(f"No Spectrometer & Blind Eye at 4500K -> Mentally Predicted Wavelength = "
          f"{perc.inferred_world['emission_wavelength_nm']:.4f} nm "
          f"(True = {unseen_4500.emission_wavelength_nm:.4f} nm)")

    cf_sync = sync_synthesis_to_causal_field(report, CausalField(dimensions=3), prefix="doc")
    cf_sync["causal_field"].step(dt=0.1)
    print(f"CausalField Sync: verified={len(report.verified_laws)}, dim_rej={len(report.dimensional_rejections)}, "
          f"falsified={len(report.falsified_claims)}, metaphors={len(report.metaphor_reflections)}, "
          f"abstract_sky_curiosity={cf_sync['abstract_sky_curiosity']:.1f}")

    print("\n" + "=" * 92)
    print("  ALL PHASE 12 LIVING CAUSAL TEXT SYNTHESIZER VERIFICATIONS COMPLETED.")
    print("=" * 92)


if __name__ == "__main__":
    run_demo()
