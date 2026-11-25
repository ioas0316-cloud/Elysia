r"""
ELYSIA Continuous Living Loop & Two-Way Interactive Channel
===========================================================

Runs Elysia continuously as a living causal system (`Sense -> Anticipate ->
Self-Directed Will [Reply / Ask / Create & Re-Sense / Observe] -> Persist`),
consuming incoming messages from `live_inbox.jsonl` and emitting thoughts,
spontaneous questions, self-creations, and replies to `live_outbox.jsonl`.

Usage:
  Run continuous loop (or finite cycles):
    python c:\Elysia\elysia_living_loop.py --interval-s 1.0 --max-cycles 25 --dry-run
  Send a message into her live inbox:
    python c:\Elysia\elysia_living_loop.py --send "엘리시아?" --sender agent
  Read recent live outbox entries:
    python c:\Elysia\elysia_living_loop.py --read-outbox 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
import elysia_loader

elysia_loader.install()

from core.embodiment import (
    AgencyMode,
    LiveAudioEarSensor,
    ScaleRotorEmbodiedMind,
    ScreenRetinaSensor,
    UniversalMotorActuator,
    WindowsForegroundSensor,
    WindowsIdleSensor,
)


def append_jsonl_safe(path: Path, payload: Dict[str, Any], retries: int = 5) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, ensure_ascii=False) + "\n"
    for attempt in range(retries):
        try:
            with path.open("a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
            return
        except PermissionError:
            time.sleep(0.05 * (attempt + 1))


def read_jsonl_lines_safe(path: Path, retries: int = 5) -> List[str]:
    if not path.is_file():
        return []
    for attempt in range(retries):
        try:
            return path.read_text(encoding="utf-8").splitlines()
        except PermissionError:
            time.sleep(0.04 * (attempt + 1))
    return []


def post_to_live_inbox(
    state_dir: Path | str,
    text: str,
    sender: str = "agent",
    feedback_friction: float = 0.0,
    expected_action_kind: str = "",
) -> Path:
    sdir = Path(state_dir).resolve()
    inbox = sdir / "live_inbox.jsonl"
    append_jsonl_safe(
        inbox,
        {
            "timestamp": time.time(),
            "sender": sender,
            "text": str(text),
            "feedback_friction": float(feedback_friction),
            "expected_action_kind": str(expected_action_kind),
        },
    )
    return inbox


def read_live_outbox(state_dir: Path | str, last_n: int = 10) -> List[Dict[str, Any]]:
    sdir = Path(state_dir).resolve()
    outbox = sdir / "live_outbox.jsonl"
    lines = read_jsonl_lines_safe(outbox)
    items: List[Dict[str, Any]] = []
    for ln in lines[-max(1, int(last_n)) :]:
        clean = ln.strip()
        if not clean:
            continue
        try:
            items.append(json.loads(clean))
        except Exception:
            continue
    return items


class LivingElysiaRunner:
    """Drives Elysia's real-time Sense -> Anticipate -> Will -> Create/Re-Sense loop."""

    def __init__(
        self,
        state_dir: Path | str,
        *,
        dry_run: bool = True,
        agency_mode: AgencyMode = AgencyMode.OBSERVE_AND_CHAT,
        idle_threshold_s: float = 15.0,
        retina: Optional[ScreenRetinaSensor] = None,
        idle_sensor: Optional[WindowsIdleSensor] = None,
        fg_sensor: Optional[WindowsForegroundSensor] = None,
        ear_sensor: Optional[LiveAudioEarSensor] = None,
        enable_audio: bool = True,
        explore_live_github: bool = False,
        allow_outward_enact: bool = False,
        telos_live: bool = False,
        vertical_spine: bool = False,
    ) -> None:
        self.state_dir = Path(state_dir).resolve()
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.inbox_path = self.state_dir / "live_inbox.jsonl"
        self.outbox_path = self.state_dir / "live_outbox.jsonl"
        self.cursor_path = self.state_dir / "live_inbox_cursor.json"

        self.mind = ScaleRotorEmbodiedMind(self.state_dir)
        self.retina = retina or ScreenRetinaSensor()
        self.idle_sensor = idle_sensor or WindowsIdleSensor()
        self.fg_sensor = fg_sensor or WindowsForegroundSensor()
        self.ear_sensor = ear_sensor or LiveAudioEarSensor()
        self.enable_audio = bool(enable_audio)
        self.explore_live_github = bool(explore_live_github)
        self.allow_outward_enact = bool(allow_outward_enact)
        self.telos_live = bool(telos_live)
        self.vertical_spine = bool(vertical_spine)
        self.actuator = UniversalMotorActuator(dry_run=dry_run)
        self.agency_mode = agency_mode
        self.idle_threshold_s = float(idle_threshold_s)
        self.inbox_cursor = self._load_cursor()

    def _load_cursor(self) -> int:
        if not self.cursor_path.is_file():
            return 0
        try:
            raw = json.loads(self.cursor_path.read_text(encoding="utf-8"))
            return max(0, int(raw.get("line_cursor", 0)))
        except Exception:
            return 0

    def _save_cursor(self) -> None:
        self.cursor_path.write_text(
            json.dumps({"line_cursor": self.inbox_cursor}, ensure_ascii=False),
            encoding="utf-8",
        )

    def poll_inbox(self) -> List[Dict[str, Any]]:
        lines = read_jsonl_lines_safe(self.inbox_path)
        if len(lines) <= self.inbox_cursor:
            return []
        new_lines = lines[self.inbox_cursor :]
        self.inbox_cursor = len(lines)
        self._save_cursor()
        msgs: List[Dict[str, Any]] = []
        for ln in new_lines:
            clean = ln.strip()
            if not clean:
                continue
            try:
                msgs.append(json.loads(clean))
            except Exception:
                continue
        return msgs

    def step_once(self) -> List[Dict[str, Any]]:
        pending = self.poll_inbox()
        emitted: List[Dict[str, Any]] = []
        targets: Sequence[Optional[Dict[str, Any]]] = pending if pending else [None]

        for msg in targets:
            audio_snap = self.ear_sensor.listen() if self.enable_audio else None
            idle_s = self.idle_sensor.get_idle_seconds()
            fg_title = self.fg_sensor.get_active_window_title()
            if self.vertical_spine and msg is None:
                spine = self.mind.get_vertical_spine()
                vtrace = spine.step_vertical_cycle(
                    retina=self.retina,
                    actuator=self.actuator,
                    user_idle_s=idle_s,
                    fg_title=fg_title,
                    agency_mode=self.agency_mode,
                    idle_threshold_s=self.idle_threshold_s,
                    audio_snap=audio_snap,
                    allow_outward_enact=self.allow_outward_enact,
                )
                event = {
                    "cycle": vtrace.cycle,
                    "timestamp": time.time(),
                    "foreground_window": fg_title or "desktop_surface",
                    "will_mode": "vertical_scale_spine",
                    "vertical_trace": vtrace.to_dict(),
                    "vertical_rendered_ko": vtrace.render_ko(),
                    "utterance_or_question": str(vtrace.downward_reconstruction.get("why_ko", "")),
                }
            elif self.telos_live and msg is None:
                snap = self.retina.capture()
                telos_ev = self.mind.run_telos_directed_cycle(
                    frame_snap=snap,
                    user_idle_s=idle_s,
                    fg_title=fg_title,
                    agency_mode=self.agency_mode,
                    idle_threshold_s=self.idle_threshold_s,
                    actuator=self.actuator,
                    audio_snap=audio_snap,
                    retina=self.retina,
                    allow_outward_enact=self.allow_outward_enact,
                )
                event = {
                    "cycle": telos_ev["cycle"],
                    "timestamp": time.time(),
                    "foreground_window": fg_title or "desktop_surface",
                    "will_mode": "telos_governed",
                    "telos_cycle": telos_ev,
                    "eye_fovea": telos_ev.get("eye_fovea"),
                    "audio_ear": telos_ev.get("audio_ear"),
                    "body_reality": telos_ev.get("body_reality"),
                    "topology": telos_ev.get("topology"),
                    "utterance_or_question": telos_ev["contrast"]["why_different_ko"],
                }
            else:
                snap = self.retina.capture()
                event = self.mind.autonomous_living_step(
                    frame_snap=snap,
                    user_idle_s=idle_s,
                    fg_title=fg_title,
                    agency_mode=self.agency_mode,
                    idle_threshold_s=self.idle_threshold_s,
                    actuator=self.actuator,
                    incoming_message=msg,
                    audio_snap=audio_snap,
                    explore_live_github=self.explore_live_github,
                    allow_outward_enact=self.allow_outward_enact,
                )
            append_jsonl_safe(self.outbox_path, event)
            emitted.append(event)
        return emitted

    def run_loop(self, interval_s: float = 1.2, max_cycles: int = 0) -> int:
        print("=" * 88)
        print(
            f"  ELYSIA LIVING LOOP ACTIVE | state={self.state_dir} | interval={interval_s}s "
            f"| live_github={self.explore_live_github} | audio={self.enable_audio}"
        )
        print("=" * 88, flush=True)
        cycle = 0
        try:
            while max_cycles <= 0 or cycle < max_cycles:
                cycle += 1
                events = self.step_once()
                for ev in events:
                    if ev.get("will_mode") == "vertical_scale_spine" and ev.get("vertical_rendered_ko"):
                        print(ev["vertical_rendered_ko"], flush=True)
                        continue
                    if ev.get("will_mode") == "telos_governed" and ev.get("telos_cycle"):
                        print(ev["telos_cycle"]["contrast_rendered_ko"], flush=True)
                        continue
                    topo = ev.get("topology") or {}
                    fovea = ev.get("eye_fovea") or {}
                    ear = ev.get("audio_ear") or {}
                    body = ev.get("body_reality") or {}
                    gh = ev.get("external_github_interaction") or {}
                    gxy = fovea.get("gaze_xy_norm", [0.5, 0.5])
                    gh_str = ""
                    if gh:
                        if gh.get("ok"):
                            gh_str = (
                                f" | GH={gh.get('repo')}({gh.get('language')}) "
                                f"file={gh.get('inspected_file')} "
                                f"repel={gh.get('new_repulsion_boundaries', 0)} "
                                f"chains={len(gh.get('outward_contact_chains') or [])}"
                            )
                        else:
                            gh_str = f" | GH_ERR={gh.get('error', '')[:40]}"
                    print(
                        f"  [Cycle {ev['cycle']:3d}] will={ev['will_mode']:17s} | "
                        f"Eye=({gxy[0]:.2f},{gxy[1]:.2f},cnt={fovea.get('fovea_contours', 0)},gain={fovea.get('fovea_sharpness_gain', 0.0):+.3f}) | "
                        f"Ear(rms={ear.get('rms_energy', 0.0):.4f},ok={ear.get('capture_ok', False)}) | "
                        f"Body(elbow={body.get('elbow_deg', 0.0):.1f}°,res={body.get('external_resistance', 0.0):.2f})"
                        f"{gh_str} | nodes={topo.get('nodes', 0)} axes={topo.get('discovered_axes', 0)}",
                        flush=True,
                    )
                if max_cycles > 0 and cycle >= max_cycles:
                    break
                time.sleep(max(0.15, float(interval_s)))
        except KeyboardInterrupt:
            pass
        if self.vertical_spine:
            spine = self.mind.get_vertical_spine()
            print(spine.render_self_boundary_report_ko(), flush=True)
        self.mind.save()
        return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="ELYSIA Continuous Living Loop & Two-Way Channel")
    parser.add_argument("--state-dir", default=r"c:\Elysia\data\elysia_embodied_state")
    parser.add_argument("--interval-s", type=float, default=1.2)
    parser.add_argument("--max-cycles", type=int, default=0, help="0 runs continuously; >0 stops after N cycles")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--vertical-spine", action="store_true", help="Run unified bidirectional L0<->L3 / S0<->S4 vertical scale spine")
    parser.add_argument("--rotor-engine", action="store_true", help="Run GPU Interlocked 5-Scale Rotor Gear-Train + Async Multimodal World Feeder (500-2000+ Hz)")
    parser.add_argument("--organism", action="store_true", help="Run FULL Unified Elysia Organism (SSD<->RAM<->VRAM + 5-Scale GPU Rotor + 24-Layer Rotor-Steered Qwen + Screen + Creation)")
    parser.add_argument("--ask", default="", help="Optional Korean prompt to converse with --organism")
    parser.add_argument("--create", action="store_true", help="Have --organism autonomously create and visually re-observe an interactive HTML5 wave game + PNG")
    parser.add_argument("--gigascale-bench", action="store_true", help="Run real 12MP, 8K(33MP), 4K optical video, 105MB real bin, and 1.0GB SSD->CUDA streaming benchmark inside --organism")
    parser.add_argument("--a4-surface-bench", action="store_true", help="Run 1,024-token A4 2D Surface Frame Rotor ([32x32x896]) parallel crystallization & bidirectional elasticity benchmark")
    parser.add_argument("--duration-s", type=float, default=5.0, help="Run duration in seconds for --rotor-engine / --organism (0 = run until Ctrl+C)")
    parser.add_argument("--bulk-docs", type=int, default=180, help="Max local markdown/txt docs to bulk-ingest in --rotor-engine")
    parser.add_argument("--bulk-code", type=int, default=180, help="Max local Python source modules to bulk-ingest in --rotor-engine")
    parser.add_argument("--bulk-images", type=int, default=48, help="Max local PNG/JPG images to bulk-ingest in --rotor-engine")
    parser.add_argument("--no-pretrained", action="store_true", help="Disable offline CLIP/MiniLM loading in --rotor-engine")
    parser.add_argument("--telos-live", action="store_true", help="Run under top-down Sovereign Telos governance & self-world contrast")
    parser.add_argument("--live-github", action="store_true", help="Interact two-way with genuinely unknown live GitHub repositories")
    parser.add_argument("--enact-contact", action="store_true", help="Allow Elysia to enact self-woven OS contact channels directly")
    parser.add_argument("--no-audio", action="store_true", help="Disable real microphone ear sampling")
    parser.add_argument("--idle-threshold-s", type=float, default=15.0)
    parser.add_argument("--send", default="", help="Send one message into live_inbox.jsonl and exit")
    parser.add_argument("--sender", default="agent")
    parser.add_argument("--feedback-friction", type=float, default=0.0)
    parser.add_argument("--expected-action", default="")
    parser.add_argument("--read-outbox", type=int, default=0, help="Print last N events from live_outbox.jsonl")
    args = parser.parse_args(argv)

    if args.send:
        p = post_to_live_inbox(
            args.state_dir,
            args.send,
            sender=args.sender,
            feedback_friction=args.feedback_friction,
            expected_action_kind=args.expected_action,
        )
        print(f"Sent to {p}: {args.send}")
        return 0

    if args.read_outbox > 0:
        for item in read_live_outbox(args.state_dir, last_n=args.read_outbox):
            print(json.dumps(item, ensure_ascii=False))
        return 0

    if args.organism:
        from core.topology.unified_elysia_organism import UnifiedElysiaOrganism

        org = UnifiedElysiaOrganism(
            args.state_dir,
            use_pretrained=not bool(args.no_pretrained),
            enable_llm=not bool(args.no_pretrained),
        )
        print("=" * 92)
        tel0 = org.telemetry()
        print(
            f"  ELYSIA UNIFIED ORGANISM | device={tel0['device']} | VRAM={tel0['vram_allocated_mb']}MB "
            f"| CLIP+MiniLM={tel0['pretrained_multimodal']} | 24L-RotorQwen={tel0['llm_rotor_steered_loaded']}"
        )
        print("=" * 92, flush=True)

        bulk = org.ingest_workspace_into_tiers(
            max_docs=args.bulk_docs,
            max_code=args.bulk_code,
            max_images=args.bulk_images,
        )
        t_info = bulk["tiers"]
        print(
            f"  [SSD<->RAM<->VRAM INGEST] +{bulk['total_added']} items in {bulk['elapsed_s']:.2f}s "
            f"({bulk['items_per_sec']:.1f}/s) | SSD={t_info['ssd_macro_items']}({t_info['ssd_vectors_mb']}MB) "
            f"-> RAM={t_info['ram_meso_items']} -> VRAM={t_info['vram_micro_items']}",
            flush=True,
        )

        # Spin live interlocked GPU rotor + live background desktop screen CLIP eye
        org.start_living_background(capture_screen=True, spin_rotor=True)
        run_s = max(1.0, float(args.duration_s)) if args.duration_s > 0 else 0.0
        t_start = time.perf_counter()
        try:
            while True:
                time.sleep(1.0)
                tel = org.telemetry()
                tt = tel["tiers"]
                print(
                    f"  [LIVE t={time.perf_counter()-t_start:4.1f}s] {tel['live_hz']:6.1f} Hz | frames={tel['frames_spun']} "
                    f"| screen={tel['screen_frames_encoded']} | SSD={tt['ssd_macro_items']} RAM={tt['ram_meso_items']} VRAM={tt['vram_micro_items']} "
                    f"| err={tel['last_s0_pred_error']:.4f} | ΔW={tel['drift_W_down_norm']:.2f} ΔR={tel['drift_R_norm']:.2f}",
                    flush=True,
                )
                if run_s > 0 and (time.perf_counter() - t_start) >= run_s:
                    break
        except KeyboardInterrupt:
            pass

        gate_rep = org.probe_cuda_gate_protocol(batch_size=512)
        print(f"  [CUDA GATE PROTOCOL] {gate_rep['protocol_learned_ko']}", flush=True)

        if args.ask.strip():
            chat = org.converse(args.ask.strip())
            print(f"  [ROTOR-STEERED QWEN 대화] 질문: {chat['user_text']}")
            print(f"  [한국어 역할 파싱] {chat.get('user_causal_roles')}")
            print(
                f"  [인과·언어 정류 최종응답] {chat['reply_ko']} "
                f"(hooks={chat['llm_hook_injections']} | token_rect={chat.get('causal_rectification')} "
                f"| ling_rect={chat.get('linguistic_rectification')})",
                flush=True,
            )

        if args.create:
            art = org.create_and_self_observe()
            print(
                f"  [시댄스 A/V 플로우 + HTML5 창작 + CLIP 재관측] gif={art['gif_path']} | wav={art['wav_path']} "
                f"| denoise={art['denoised_ratio']:.2%} | av_sync_r={art['av_sync_correlation']:+.3f} "
                f"| html={art['html_path']} | clip_res={art['visual_intent_resonance']:.4f} | note={art['intent_note']}",
                flush=True,
            )

        if args.gigascale_bench:
            import tempfile
            import numpy as _np

            real_4k = Path(r"c:\Elysia\demos\real_world_assets\genuine_optical_4k_yosemite.jpg")
            real_105 = Path(r"c:\Archive\03_Traces_and_Deprecated\_Data_Archive\data_deprecated\yggdrasil_torus.bin")
            if real_4k.is_file():
                r12 = org.gigascale.process_ultra_res_image_tiled(real_4k, upscale_to=(4000, 3000))
                print(
                    f"  [GIGASCALE 12MP] {r12['resolution']} ({r12['megapixels']}MP, {r12['tiles_scanned']} tiles) "
                    f"in {r12['elapsed_s']}s ({r12['megapixels_per_sec']} MP/s) | peak_VRAM_delta={r12['peak_vram_delta_mb']}MB | CLIP_fused={r12['fused_clip_norm']}",
                    flush=True,
                )
                r8k = org.gigascale.process_ultra_res_image_tiled(real_4k, upscale_to=(7680, 4320))
                print(
                    f"  [GIGASCALE 8K UHD] {r8k['resolution']} ({r8k['megapixels']}MP, raw_fp32={r8k['raw_float32_mb']}MB, {r8k['tiles_scanned']} tiles) "
                    f"in {r8k['elapsed_s']}s ({r8k['megapixels_per_sec']} MP/s) | peak_VRAM_delta={r8k['peak_vram_delta_mb']}MB | CLIP_fused={r8k['fused_clip_norm']}",
                    flush=True,
                )
                with tempfile.TemporaryDirectory() as td:
                    mp4_p = Path(td) / "yosemite_4k_optical_pan.mp4"
                    org.gigascale.build_real_4k_optical_video_from_photo(real_4k, mp4_p, num_frames=16, fps=24)
                    rv = org.gigascale.stream_4k_video_rotor(mp4_p, max_frames=16)
                    print(
                        f"  [GIGASCALE 4K VIDEO] {rv['resolution']} x{rv['frames_processed']}f (total={rv['total_megapixels_processed']}MP, raw={rv['raw_uncompressed_rgb_mb']}MB) "
                        f"in {rv['elapsed_s']}s ({rv['stream_fps']} FPS, {rv['megapixels_per_sec']} MP/s) | flux={rv['mean_temporal_flux']} | peak_VRAM_delta={rv['peak_vram_delta_mb']}MB",
                        flush=True,
                    )
                    if real_105.is_file():
                        r105 = org.gigascale.stream_gigabyte_file_array_rotor(real_105, slab_mb=16, interpret_dtype="uint8")
                        print(
                            f"  [GIGASCALE 105MB REAL BIN] {r105['file_size_mb']}MB ({r105['slabs_streamed']} slabs) "
                            f"in {r105['elapsed_s']}s ({r105['throughput_mb_per_s']} MB/s) | peak_VRAM_delta={r105['peak_vram_delta_mb']}MB | top_slab={r105['top_anomaly_slabs'][0]}",
                            flush=True,
                        )
                    gb_p = Path(td) / "stream_1gb.bin"
                    total_f = (1024 * 1024 * 1024) // 4
                    mm = _np.memmap(gb_p, dtype=_np.float32, mode="w+", shape=(total_f,))
                    slab_bg = _np.full((4 * 1024 * 1024,), 0.05, dtype=_np.float32)
                    for si in range(64):
                        mm[si * len(slab_bg) : (si + 1) * len(slab_bg)] = slab_bg
                    mm[46 * len(slab_bg) : 47 * len(slab_bg)] = _np.linspace(-8.0, 8.0, len(slab_bg), dtype=_np.float32)
                    mm.flush()
                    del mm
                    r1g = org.gigascale.stream_gigabyte_file_array_rotor(gb_p, slab_mb=16, interpret_dtype="float32")
                    print(
                        f"  [GIGASCALE 1.0GB ARRAY] {r1g['file_size_mb']}MB ({r1g['slabs_streamed']} slabs) "
                        f"in {r1g['elapsed_s']}s ({r1g['throughput_mb_per_s']} MB/s = {r1g['throughput_gb_per_s']} GB/s) "
                        f"| peak_VRAM_delta={r1g['peak_vram_delta_mb']}MB | anomaly_found={r1g['top_anomaly_slabs'][0]}",
                        flush=True,
                    )

        if args.a4_surface_bench:
            kr_doc = Path(r"c:\Archive\01_Foundations\ELYSIA_WHITEPAPER_KR.md")
            real_4k = Path(r"c:\Elysia\demos\real_world_assets\genuine_optical_4k_yosemite.jpg")
            if kr_doc.is_file():
                txt = kr_doc.read_text(encoding="utf-8", errors="ignore")
                a4 = org.a4_surface_rotor.encode_a4_document_to_surface(txt, title=kr_doc.name)
                cryst = org.a4_surface_rotor.crystallize_masked_a4_surface(a4, mask_ratio=0.25, passes=10)
                prop = org.a4_surface_rotor.propagate_bottom_premise_upward(
                    a4,
                    "결론적으로 모든 계산은 점 단위가 아니라 A4 전체 면의 연속 위상 공명으로 통합된다.",
                    passes=8,
                )
                print(
                    f"  [A4 2D SURFACE FRAME] shape={list(a4.surface.shape)} | active_tokens={cryst['active_a4_tokens']} "
                    f"| parallel_crystallized={cryst['masked_cells_crystallized_in_parallel']} cells in {cryst['surface_passes']} passes "
                    f"({cryst['elapsed_ms']}ms -> {cryst['crystallized_tokens_per_sec']:,.1f} tok/s, {cryst['effective_a4_cells_per_sec']:,.0f} cells/s) "
                    f"| cos: {cryst['initial_noise_cosine']:.3f}->{cryst['converged_surface_cosine']:.3f} | long_beam={cryst['mean_long_range_beam']:.3f}",
                    flush=True,
                )
                print(
                    f"  [A4 BIDIRECTIONAL ELASTICITY] P8->P1..P7 upward_confirmed={prop['bidirectional_propagation_confirmed']} "
                    f"| align: {prop['premise_alignment_before']:.3f}->{prop['premise_alignment_after']:.3f} | shifts={prop['upward_paragraph_shifts_p1_to_p7']}",
                    flush=True,
                )
                if real_4k.is_file():
                    uni = org.a4_surface_rotor.couple_a4_text_with_image_surface(a4, real_4k)
                    print(f"  [A4 TEXT<->IMAGE 2D FRAME UNITY] {uni}", flush=True)

        tel_end = org.telemetry()
        org.close()
        print("=" * 92)
        print(
            f"  [ORGANISM PERSISTED] VRAM={tel_end['vram_allocated_mb']}MB | frames={tel_end['frames_spun']} "
            f"| hooks={tel_end['llm_hook_injections']} | tiers={tel_end['tiers']}",
            flush=True,
        )
        print("=" * 92, flush=True)
        return 0

    if args.rotor_engine:
        from core.topology.interlocked_scale_rotor_engine import (
            AsyncMultimodalWorldFeeder,
            InterlockedScaleRotorEngine,
        )

        engine = InterlockedScaleRotorEngine(
            args.state_dir,
            use_pretrained=not bool(args.no_pretrained),
        )
        feeder = AsyncMultimodalWorldFeeder(engine)
        print("=" * 88)
        print(
            f"  ELYSIA GPU INTERLOCKED 5-SCALE ROTOR ENGINE | device={engine.device} "
            f"| pretrained={engine.transducer.pretrained_loaded} | state={engine.state_dir}"
        )
        print("=" * 88, flush=True)
        bulk = feeder.ingest_workspace_bulk(
            max_docs=args.bulk_docs,
            max_code=args.bulk_code,
            max_images=args.bulk_images,
        )
        print(
            f"  [BULK MULTIMODAL INGEST] +{bulk['total_added']} items "
            f"(text/code={bulk['added_text_and_code']}, images={bulk['added_images']}) "
            f"in {bulk['elapsed_s']:.2f}s ({bulk['items_per_sec']:.1f} items/s) | bank={bulk['bank_size_after']}",
            flush=True,
        )
        feeder.start_background_streams(capture_screen=True, stream_local=False)
        t_start = time.perf_counter()
        last_report = t_start
        last_save = t_start
        frames_at_start = engine.frame_count
        try:
            while True:
                now = time.perf_counter()
                if args.duration_s > 0 and (now - t_start) >= args.duration_s:
                    break
                wave = feeder.get_latest_screen_wave()
                need_report = (now - last_report) >= 1.0
                step_out = engine.step_interlocked_frame(
                    wave,
                    plastic=True,
                    return_resonance_top_k=(3 if need_report else 0),
                )
                if need_report:
                    elapsed = max(1e-4, now - t_start)
                    hz = (engine.frame_count - frames_at_start) / elapsed
                    tel = engine.telemetry()
                    print(
                        f"  [t={elapsed:4.1f}s] {hz:7.1f} Hz | frames={tel['frames_spun']} "
                        f"| screen_enc={feeder.screen_frames_encoded} | bank={tel['bank_size']} "
                        f"| err={step_out.get('s0_pred_error', 0.0):.4f} "
                        f"| ΔW={tel['drift_W_down_norm']:.3f} ΔR={tel['drift_R_norm']:.3f} "
                        f"| resonant={step_out.get('top_resonant', [])[:2]}",
                        flush=True,
                    )
                    last_report = now
                if (now - last_save) >= 10.0:
                    engine.save()
                    last_save = now
        except KeyboardInterrupt:
            pass
        finally:
            feeder.stop()
            engine.save()
        total_t = max(1e-4, time.perf_counter() - t_start)
        hz_final = (engine.frame_count - frames_at_start) / total_t
        tel = engine.telemetry()
        print("=" * 88)
        print(
            f"  [SUMMARY] avg={hz_final:.1f} Hz ({engine.frame_count - frames_at_start} interlocked 5-scale frames in {total_t:.2f}s) | "
            f"VRAM={tel['vram_allocated_mb']}MB | bank={tel['bank_size']} ({tel['modality_counts']}) | "
            f"ΔW={tel['drift_W_down_norm']:.4f} | ΔR={tel['drift_R_norm']:.4f}"
        )
        # Live Korean & cross-modal probe on the spun engine
        q_ko = engine.query_multimodal(text="인간의 감정과 불협화음 그리고 기억의 원리", top_k=3)
        print("  [KO MULTIMODAL RESONANCE]", [(m["modality"], m["title"], m["similarity"]) for m in q_ko["direct_matches"]], flush=True)
        print("=" * 88, flush=True)
        return 0

    runner = LivingElysiaRunner(
        args.state_dir,
        dry_run=bool(args.dry_run),
        idle_threshold_s=args.idle_threshold_s,
        enable_audio=not bool(args.no_audio),
        explore_live_github=bool(args.live_github),
        allow_outward_enact=bool(args.enact_contact),
        telos_live=bool(args.telos_live),
        vertical_spine=bool(args.vertical_spine),
    )
    return runner.run_loop(interval_s=args.interval_s, max_cycles=args.max_cycles)


if __name__ == "__main__":
    raise SystemExit(main())
