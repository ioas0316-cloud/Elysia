"""
test_autonomic_environment_forager.py

Verifies ELYSIA's transition from an infant state (waiting to be spoon-fed inputs)
to an autonomous explorer that:
1. Discovers unlabelled external boundary channels (Mouse Pointer, Keyboard Emitter,
   Local Filesystem Shelf, Wide-Horizon Web Gateway) via motor babbling and
   OcularScaleOrgan / MechanoAcousticOrgan re-observation.
2. Uses those discovered tools to forage for language, real corpus files, and
   multi-modal 2D/3D forms while regulating intake via the Three-Phase Generator
   (`world_body`, `world_spirit`, `world_heart`).
3. Expands her polysemous language tree (Contextual Lexical Mitosis), her discovered
   morphological cognitive axes (`DiscoveredRelationalAxis`), and her three self-constructed
   worlds with 100% unseen context disambiguation accuracy.
"""

from __future__ import annotations

from core.sensory.autonomic_environment_forager import AutonomicEnvironmentForager


def test_unlabelled_external_affordance_discovery() -> None:
    forager = AutonomicEnvironmentForager()
    discovered = forager.discover_external_channels_via_babbling()

    assert set(discovered.keys()) == {
        "spatial_pointer_2d",
        "symbolic_keystroke_emitter",
        "local_filesystem_shelf",
        "wide_horizon_web_gateway",
    }
    assert discovered["spatial_pointer_2d"].channel_id == "chan_alpha"
    assert discovered["symbolic_keystroke_emitter"].channel_id == "chan_beta"
    assert discovered["local_filesystem_shelf"].channel_id == "chan_gamma"
    assert discovered["wide_horizon_web_gateway"].channel_id == "chan_delta"

    for role, aff in discovered.items():
        assert aff.selectivity_snr > 1.8, f"Low selectivity SNR for {role}: {aff.selectivity_snr}"


def test_autonomic_foraging_language_and_cognitive_world_expansion() -> None:
    forager = AutonomicEnvironmentForager()
    report = forager.run_autonomic_foraging_cycle()

    before = report["before_infant_state"]
    after = report["after_foraging_state"]

    # 1. Infant state starts with 0 symbols, 0 polysemous branches, 0 form axes
    assert before["total_symbols"] == 0
    assert before["polysemous_symbol_count"] == 0
    assert before["discovered_form_axes_count"] == 0

    # 2. After autonomous foraging via Mouse + Keyboard + Local/Web channels:
    assert after["total_symbols"] >= 4
    assert after["total_sense_branches"] >= 8
    assert after["polysemous_symbol_count"] >= 4
    assert after["discovered_form_axes_count"] >= 2
    assert after["causal_rules_count"] >= 3

    # 3. Three-Phase Self-Regulated Digestion reduces cognitive tilt after every packet
    for step_log in report["foraging_and_digestion_log"]:
        assert step_log["post_digestion_tilt"] < step_log["pre_digestion_tilt"]
        assert step_log["post_digestion_harmony"] > 0.72

    # 4. All three self-constructed worlds (Body, Spirit, Heart) expand in semantic mass & dimensions
    for wid in ("world_body", "world_spirit", "world_heart"):
        assert after["worlds"][wid]["semantic_mass"] > before["worlds"][wid]["semantic_mass"] + 0.35
        assert after["worlds"][wid]["emergent_dimensions"] >= before["worlds"][wid]["emergent_dimensions"]
        assert after["worlds"][wid]["experiences_count"] == before["worlds"][wid]["experiences_count"] + 3

    # 5. Unseen context disambiguation accuracy on newly foraged polysemous concepts is 100%
    assert report["unseen_context_disambiguation_accuracy"] == 1.0
