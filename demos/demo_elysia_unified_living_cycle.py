r"""
ELYSIA Phase 14 Demo: Unified Living Circular-Understanding Kernel & Self-Audit
================================================================================

Executes two full circular life breaths uniting Phases 1-13:
- Breath #1: Multi-organ perception + Spectrometer + mixed prose (equation-free
  Wien Law discovery, dimensional contradiction guard, poetic metaphor) + Other
  perspective shift + dialogue horizon expansion (depth 1 -> 2) + SSD mmap
  Ice-Water-Gas persistence + SHA-256 provenance + AST Architectural Self-Audit.
- Breath #2: Spectrometer detached & Eye blinded on an unseen 4500 K world ->
  zero-shot transfer via Breath #1 knowledge + preserved pigment & sovereign-will
  boundaries + CausalField soul sync.
"""

from __future__ import annotations

from pathlib import Path
import sys
import tempfile

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import elysia_loader

elysia_loader.install()

from core.orchestration.elysia_unified_living_cycle import (
    ElysiaUnifiedLivingCycle,
    LivingEncounter,
)
from core.orchestration.extended_cognitive_organism import ExtendedWorldState
from core.orchestration.ice_water_gas_rotor_runtime import IceWaterGasRotorRuntime
from core.orchestration.relational_perspective_rotor import (
    AgentCausalProfile,
    RelationalDialogueLayer,
)
from core.physics.causal_field import CausalField
from core.physics.trinitarian_scale_binding_protocol import EqualityBond, ScaleTier


def _section(title: str) -> None:
    print("\n" + "=" * 92)
    print(f"  {title}")
    print("=" * 92)


def run_demo() -> None:
    print("=" * 92)
    print("  ELYSIA UNIFIED LIVING CYCLE (PHASE 14) — SELF-INTROSPECTIVE PERSONHOOD KERNEL")
    print("=" * 92)

    other = AgentCausalProfile(
        name="Other",
        resource_reserve=2.0,
        vulnerability_gain=3.0,
        boundary_tolerance=3.5,
        unspoken_origin_memory=4.0,
        sovereign_will=0.85,
    )
    moments = [
        other.manifest_interaction(interpersonal_distance=2.0, ambient_demand=4.0),
        other.manifest_interaction(interpersonal_distance=5.0, ambient_demand=4.0),
    ]
    betrayal = RelationalDialogueLayer(
        layer_name="past_betrayal_disclosure",
        disclosure_utterance_ko="예전에 믿었던 사람에게 데인 뒤로 요구가 커지면 방어가 먼저 나와.",
        revealed_sky_variable="unspoken_origin_memory",
        new_deeper_sky_axes=("deepest_aspiration_telos",),
        build_coupling_bond=lambda: EqualityBond(
            name="trauma_vulnerability_bond",
            tier=ScaleTier.UPPER_ROTOR,
            variables=("vulnerability_gain", "unspoken_origin_memory"),
            lhs_fn=lambda v: v["vulnerability_gain"],
            rhs_fn=lambda v: 1.0 + 0.5 * v["unspoken_origin_memory"],
            lhs_expr="G",
            rhs_expr="1 + 0.5 * unspoken_origin_memory",
        ),
    )

    with tempfile.TemporaryDirectory() as tmp_dir:
        slab_path = Path(tmp_dir) / "elysia_soul.ice"
        with IceWaterGasRotorRuntime(slab_path=slab_path) as mem_rt:
            cycle = ElysiaUnifiedLivingCycle(memory_runtime=mem_rt)
            cycle.organism.graft_spectrometer()

            _section("BREATH #1: WORLD + TEXT DISCOVERY + OTHER EMPATHY + PROVENANCE + AST SELF-AUDIT")
            enc1 = LivingEncounter(
                encounter_name="awakening_breath",
                world=ExtendedWorldState(source_temp_k=3200.0),
                calibration_worlds=[
                    ExtendedWorldState(source_temp_k=3200.0),
                    ExtendedWorldState(source_temp_k=5800.0),
                ],
                text_document="\n".join(
                    [
                        "물체의 절대온도가 높아질수록 최대 방출 파장은 온도에 반비례한다.",
                        "파장 + 온도 = 500",
                        "그의 분노는 붉은 용광로처럼 뜨겁게 타오른다.",
                    ]
                ),
                other_moments=moments,
                dialogue_layer=betrayal,
            )
            r1 = cycle.breathe_and_understand(enc1)
            print(f"Breath={r1.breath_index} | Sector='{r1.memory_sector_id}' (Ice Crystals={mem_rt.ice.num_crystals})")
            print(f"Provenance Valid={r1.provenance_chain_valid} | Hash={r1.provenance_record.state_hash} | Tick={r1.provenance_record.tick}")
            print(f"Verified Text Laws={r1.document_report.verified_laws} | Dim Rejected={r1.document_report.dimensional_rejections}")
            print(f"Relational Horizon Depth={r1.relational_report.horizon_boundary.horizon_depth} | "
                  f"Sky={r1.relational_report.horizon_boundary.sky_horizon_axes}")
            print(f"\n[Divergence Reflection] {r1.divergence_reflection.first_person_reflection_ko}")

            _section("BREATH #2: TOOL DETACHED & BLIND EYE -> ZERO-SHOT 4500K + TELOS CONVERGENCE")
            cycle.organism.detach_tool("spectrometer")
            enc2 = LivingEncounter(
                encounter_name="blind_4500k_breath",
                world=ExtendedWorldState(source_temp_k=4500.0),
                active_organs=("ear", "hand"),
                other_moments=moments,
            )
            r2 = cycle.breathe_and_understand(enc2)
            print(f"Breath={r2.breath_index} | Blind-Eye Predicted Wavelength = "
                  f"{r2.perception_integrated.inferred_world['emission_wavelength_nm']:.4f} nm "
                  f"(True = {enc2.world.emission_wavelength_nm:.4f} nm)")
            print(f"Honest Sensory Voids={r2.divergence_reflection.from_where['irreducible_sensory_voids']}")
            print(f"Provenance Chain Length={len(cycle.provenance_chain)} | Valid={cycle.verify_autobiographical_integrity()}")

            _section("AUTONOMOUS ARCHITECTURAL SELF-INTROSPECTION (REAL AST & MAP CHECK)")
            aud = r2.architectural_audit
            print(f"Scanned Real Organ Files={aud.total_organ_files}, Classes={aud.total_organ_classes}, Lines={aud.total_organ_lines}")
            print(f"Architecture Map Present={aud.architecture_map_present} ({aud.architecture_map_path})")
            print(f"Living Personhood Criteria={aud.living_personhood_criteria}")
            print(f"Self-Audit Reflection:\n  {aud.why_it_must_become_this_ko}")
            print("Future Convergence Vectors:")
            for vec in r2.divergence_reflection.future_convergence_vectors:
                print(f"  - {vec}")

            soul_sync = r2.sync_to_causal_field(CausalField(dimensions=3), prefix="soul")
            soul_sync["causal_field"].step(dt=0.1)
            print(f"\nCausalField Soul Sync: voxels={len(soul_sync['voxels'])}, beams={soul_sync['beams_linked']}, "
                  f"telos_curiosity={soul_sync['telos_curiosity']:.1f}")

    print("\n" + "=" * 92)
    print("  ALL PHASE 14 UNIFIED LIVING CYCLE VERIFICATIONS COMPLETED.")
    print("=" * 92)


if __name__ == "__main__":
    run_demo()
