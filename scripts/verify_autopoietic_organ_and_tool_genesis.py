"""
verify_autopoietic_organ_and_tool_genesis.py

Standalone verification and 6-panel dashboard for ELYSIA's Autopoietic Organ &
External Tool/CAD/Protocol Principle Inversion Architecture:
1. Code AST introspection & SafeASTSandbox verification
2. Missing temporal/formant speech organ diagnosis -> MDL falsification -> live synthesis
3. Blind external CAD operator inversion (`op_k0..op_k4`) -> standalone NumPy SDF kernel -> Unplugged creation
4. Opaque digital protocol inversion (`xor` & `add_mod_256`) -> live Python codecs
5. JSON cold-reload recompilation & CausalField L8..L14 projection

Generates:
  `c:/Elysia/docs/assets/autopoietic_organ_tool_genesis_inspection.png`
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "elysia_mpl"))

sys.path.insert(0, r"c:\Elysia")
import elysia_loader

elysia_loader.install()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from core.sensory.autopoietic_organ_genesis import (
    SafeASTSandbox,
    run_autopoietic_genesis_session,
)
from core.sensory.epistemic_knowledge_spine import EpistemicKnowledgeSpine
from simulators.opaque_external_tools import OpaquePacketProtocol


def main() -> int:
    out_png = Path(r"c:\Elysia\docs\assets\autopoietic_organ_tool_genesis_inspection.png")
    out_png.parent.mkdir(parents=True, exist_ok=True)

    print("[1/2] Running Autopoietic Organ & Tool Principle Inversion (blind_mode=True)...")
    rep = run_autopoietic_genesis_session(blind_mode=True)

    print("[2/2] Verifying JSON Cold-Reload & Live Recompilation from Saved Source Code...")
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt = Path(tmpdir) / "autopoietic_spine.json"
        rep.spine.save_json(ckpt)
        reloaded = EpistemicKnowledgeSpine.load_json(ckpt)
        xor_ns = SafeASTSandbox.compile_and_load(reloaded.synthesized_organs["codec_channel_a"].source_code)
        sum_ns = SafeASTSandbox.compile_and_load(reloaded.synthesized_organs["codec_channel_b"].source_code)
        test_msg = b"ELYSIA-PRINCIPLE-INVERSION"
        assert OpaquePacketProtocol("xor").accepts(xor_ns["encode_packet"](test_msg))
        assert OpaquePacketProtocol("add_mod_256").accepts(sum_ns["encode_packet"](test_msg))

    aud = rep.auditory
    cad = rep.cad
    px = rep.protocol_xor
    ps = rep.protocol_sum

    print("\n=== AUTOPOIETIC GENESIS SUMMARY ===")
    for s in rep.code_summaries:
        print(f"Code Introspected     : {s.module_name} (classes={len(s.classes)}, funcs={len(s.functions)}, ast={s.ast_nodes}, prims={s.numpy_dsp_primitives[:6]})")
    print(f"Auditory Baseline Fail: {aud.diagnosis.existing_organ_name} -> acc={aud.diagnosis.overall_accuracy:.3f}, time_rev_pair={aud.diagnosis.time_reversed_pair_accuracy:.3f}")
    for f in aud.falsified_candidates:
        print(f"Auditory Falsified    : {f['name']} -> unseen={f.get('unseen_acc', 0.0):.3f}, rev_pair={f.get('time_reversed_pair_acc', 0.0):.3f}")
    print(f"Auditory Self-Built   : {aud.winning_organ_id} (AST={aud.ast_node_count}, LOO={aud.train_loo_accuracy:.3f}, Unseen+35Hz={aud.unseen_test_accuracy:.3f})")
    print(f"CAD Blind Roles Found : {cad.discovered_op_roles}")
    print(f"CAD Kernel Parity     : IoU={cad.heldout_voxel_iou:.4f}, VolErr={cad.heldout_volume_rel_error:.4f}, Ocular_L2={cad.optical_reobservation_l2_corr:.4f}")
    print(f"CAD Unplugged Design  : stable={cad.unplugged_design_summary['stable_under_gravity']}, margin={cad.unplugged_design_summary['stability_margin']:+.3f}, cavity={cad.unplugged_design_summary['hollow_cavity_voxels']} vox")
    print(f"Protocol Channel A    : hdr=0x{px.discovered_header_hex}, len_idx={px.discovered_len_index}, law={px.discovered_checksum_law}[{px.discovered_checksum_start_idx}:], acc/rt/tamper={px.unseen_encode_accept_rate:.0%}/{px.unseen_roundtrip_accuracy:.0%}/{px.tamper_rejection_rate:.0%}")
    print(f"Protocol Channel B    : hdr=0x{ps.discovered_header_hex}, len_idx={ps.discovered_len_index}, law={ps.discovered_checksum_law}[{ps.discovered_checksum_start_idx}:], acc/rt/tamper={ps.unseen_encode_accept_rate:.0%}/{ps.unseen_roundtrip_accuracy:.0%}/{ps.tamper_rejection_rate:.0%}")
    print(f"CausalField Topology  : {len(rep.causal_field.voxels)} voxels, {len(rep.causal_field.beams)} beams, open_voids={len(rep.spine.get_open_voids())}")

    print("\n--- Self-Synthesized Auditory Organ Source (Excerpt) ---")
    print("\n".join(aud.source_code.splitlines()[:12]))
    print("--- Self-Synthesized Protocol Codec Source (Channel A) ---")
    print("\n".join(px.synthesized_codec_source.splitlines()[:10]))

    # Render 6-panel dark-theme inspection figure
    plt.style.use("dark_background")
    fig, axes = plt.subplots(2, 3, figsize=(17.5, 10.2), dpi=135)
    fig.patch.set_facecolor("#0f1419")

    # Panel 1: Missing Organ Diagnosis & Empirical Falsification Loop
    ax1 = axes[0, 0]
    labels_1 = [
        "Existing\nMechanoTap",
        "Cand 1\nStatic FFT",
        "Cand 2\nRMS Env",
        "Mounted\nSTFT Formant",
    ]
    overall_vals = [
        aud.diagnosis.overall_accuracy,
        aud.falsified_candidates[0].get("unseen_acc", 0.0),
        aud.falsified_candidates[1].get("unseen_acc", 0.0),
        aud.unseen_test_accuracy,
    ]
    rev_vals = [
        aud.diagnosis.time_reversed_pair_accuracy,
        aud.falsified_candidates[0].get("time_reversed_pair_acc", 0.0),
        aud.falsified_candidates[1].get("time_reversed_pair_acc", 0.0),
        1.0,
    ]
    x1 = np.arange(len(labels_1))
    ax1.bar(x1 - 0.18, overall_vals, width=0.34, color="#48cae4", label="4-Class Acc (Unseen +35Hz)")
    ax1.bar(x1 + 0.18, rev_vals, width=0.34, color="#ff6b6b", label="Time-Reversed Pair (0 vs 1)")
    ax1.axhline(0.92, color="#80ed99", linestyle="--", linewidth=1.3, label="Falsification Bar (0.92)")
    ax1.set_xticks(x1)
    ax1.set_xticklabels(labels_1, fontsize=8.5)
    ax1.set_ylim(0.0, 1.15)
    ax1.set_title("1. Missing Auditory Organ: Diagnosis & Falsification", fontsize=10.5, fontweight="bold", pad=8)
    ax1.legend(fontsize=7.5, loc="upper left")
    ax1.grid(axis="y", alpha=0.22)

    # Panel 2: Blind CAD Kernel Parity (External vs Self-Synthesized Python SDF)
    ax2 = axes[0, 1]
    ext_u8 = np.asarray(cad.external_reference_rgb, dtype=np.uint8)
    self_u8 = np.asarray(cad.self_kernel_reference_rgb, dtype=np.uint8)
    pair_img = np.concatenate([ext_u8, self_u8], axis=1)
    ax2.imshow(pair_img)
    ax2.axvline(ext_u8.shape[1], color="#fee440", linestyle="--", linewidth=1.4)
    ax2.set_xticks([])
    ax2.set_yticks([])
    ax2.set_title(
        f"2. Blind CAD Inversion: Ext Tool (L) vs Self Code (R)\n"
        f"IoU={cad.heldout_voxel_iou:.3f} | VolErr={cad.heldout_volume_rel_error:.3f} | Ocular_L2={cad.optical_reobservation_l2_corr:.3f}",
        fontsize=9.8,
        fontweight="bold",
        pad=8,
    )
    ax2.text(
        0.02, 0.04,
        "op_k0=Box, op_k1=ExtrudeZ, op_k2=Union, op_k3=Diff, op_k4=Scale",
        transform=ax2.transAxes,
        fontsize=7.8,
        bbox=dict(boxstyle="round", facecolor="#141a21", alpha=0.88),
    )

    # Panel 3: Empirical Gravity Tipping Law
    ax3 = axes[0, 2]
    shifts = [r["top_shift_x"] for r in cad.stability_boundary_probe]
    margins = [r["stability_margin"] for r in cad.stability_boundary_probe]
    colors3 = ["#80ed99" if m >= 0 else "#ff595e" for m in margins]
    ax3.plot(shifts, margins, color="#90e0ef", linewidth=2.0, zorder=2)
    ax3.scatter(shifts, margins, c=colors3, s=85, zorder=3, edgecolors="white", linewidths=0.9)
    ax3.axhline(0.0, color="#fee440", linestyle="--", linewidth=1.4, label="Tipping Boundary (margin=0)")
    ax3.set_xlabel("Top Block Offset dx (Probe Parameter)")
    ax3.set_ylabel("Base Support COM Margin")
    ax3.set_title("3. Inverted Gravity Tipping Stability Boundary", fontsize=10.5, fontweight="bold", pad=8)
    ax3.legend(fontsize=8, loc="upper right")
    ax3.grid(alpha=0.22)

    # Panel 4: UNPLUGGED Design (External Tool Disconnected) + Internal Hollow Bore Cross-Section
    ax4 = axes[1, 0]
    des_u8 = np.asarray(cad.unplugged_design_rgb, dtype=np.uint8)
    # Cross-section XZ plane at y = 12 (24x24 scaled cleanly to match 96x96)
    slice_occ = np.rot90(cad.unplugged_design_occ[:, 12, :], k=1).astype(np.uint8) * 215
    slice_rgb = np.stack([slice_occ, (slice_occ * 0.82).astype(np.uint8), (slice_occ * 0.65).astype(np.uint8)], axis=-1)
    # Nearest neighbor crisp upscale to 96x96
    slice_96 = np.repeat(np.repeat(slice_rgb, 4, axis=0), 4, axis=1)
    unp_comb = np.concatenate([des_u8, slice_96], axis=1)
    ax4.imshow(unp_comb)
    ax4.axvline(des_u8.shape[1], color="white", linestyle=":", linewidth=1.2)
    ax4.set_xticks([])
    ax4.set_yticks([])
    ds = cad.unplugged_design_summary
    ax4.set_title(
        f"4. UNPLUGGED Creation (3D Render + Hollow XZ Slice)\n"
        f"Stable={ds['stable_under_gravity']} | Margin={ds['stability_margin']:+.2f} | Cavity={ds['hollow_cavity_voxels']} vox",
        fontsize=9.8,
        fontweight="bold",
        pad=8,
    )

    # Panel 5: Opaque Digital Protocol Principle Inversion (XOR vs ADD_MOD_256)
    ax5 = axes[1, 1]
    metrics_5 = ["Encode\nAccept", "Roundtrip\nDecode", "Tamper\nReject"]
    xor_v = [px.unseen_encode_accept_rate, px.unseen_roundtrip_accuracy, px.tamper_rejection_rate]
    sum_v = [ps.unseen_encode_accept_rate, ps.unseen_roundtrip_accuracy, ps.tamper_rejection_rate]
    x5 = np.arange(len(metrics_5))
    ax5.bar(x5 - 0.18, xor_v, width=0.34, color="#f77f00", label=f"Ch A: 0x{px.discovered_header_hex} ({px.discovered_checksum_law})")
    ax5.bar(x5 + 0.18, sum_v, width=0.34, color="#00f5d4", label=f"Ch B: 0x{ps.discovered_header_hex} ({ps.discovered_checksum_law})")
    ax5.set_xticks(x5)
    ax5.set_xticklabels(metrics_5, fontsize=9)
    ax5.set_ylim(0.0, 1.25)
    ax5.set_title("5. Opaque Wire Protocol Codec Synthesis", fontsize=10.5, fontweight="bold", pad=8)
    ax5.legend(fontsize=8, loc="lower right")
    ax5.grid(axis="y", alpha=0.22)

    # Panel 6: AST Parsimony (MDL) of Synthesized Organs vs Real Core & CausalField
    ax6 = axes[1, 2]
    org_names = [
        "STFT Ear\n(Synth)",
        "CAD Kernel\n(Synth)",
        "Codec XOR\n(Synth)",
        "Codec SUM\n(Synth)",
    ]
    ast_counts = [
        aud.ast_node_count,
        cad.synthesized_ast_nodes,
        px.ast_node_count,
        ps.ast_node_count,
    ]
    y6 = np.arange(len(org_names))
    ax6.barh(y6, ast_counts, height=0.52, color=["#48cae4", "#80ed99", "#f77f00", "#00f5d4"])
    ax6.set_yticks(y6)
    ax6.set_yticklabels(org_names, fontsize=8.8)
    ax6.invert_yaxis()
    ax6.set_xlabel("AST Node Count (MDL Structural Size)")
    ax6.set_title(
        f"6. Self-Built AST Organs & CausalField ({len(rep.causal_field.voxels)}V / {len(rep.causal_field.beams)}B)",
        fontsize=10.2,
        fontweight="bold",
        pad=8,
    )
    for idx_v, val in enumerate(ast_counts):
        ax6.text(val + 18, idx_v, str(val), va="center", fontsize=8.5, color="white")
    ax6.set_xlim(0, max(ast_counts) * 1.18)
    ax6.grid(axis="x", alpha=0.22)

    fig.suptitle(
        "ELYSIA Autopoietic Organ Genesis & Blind External Tool/CAD/Protocol Principle Inversion",
        fontsize=13.5,
        fontweight="bold",
        y=0.985,
    )
    plt.tight_layout(rect=[0, 0.01, 1, 0.955])
    fig.savefig(out_png, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"\nSaved inspection figure -> {out_png}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
