r"""
ELYSIA Floating Scale-Rotor Desktop Companion (Lumi / Neuro-sama Embodiment)
============================================================================

Always-on-top translucent desktop avatar powered strictly by Elysia's unified
Scale-Rotor causal field (`ScreenRetinaSensor` + `WindowsIdleSensor` +
`UniversalMotorActuator` + `ScaleRotorEmbodiedMind`, with ZERO external LLM APIs
and ZERO per-site `if/else` scripts).

Usage:
  Interactive avatar UI:
    python c:\Elysia\elysia_desktop_companion.py
  Headless verification ticks:
    python c:\Elysia\elysia_desktop_companion.py --headless-ticks 3 --dry-run
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import tkinter as tk
from typing import Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
import elysia_loader

elysia_loader.install()

from core.embodiment import (
    AgencyMode,
    ScaleRotorEmbodiedMind,
    ScreenFrameSnapshot,
    ScreenRetinaSensor,
    UniversalMotorActuator,
    WindowsForegroundSensor,
    WindowsIdleSensor,
)


def _chroma_to_hex(red: float, blue: float, yellow: float) -> str:
    """Maps chromatic (Red=Flux, Blue=Order, Yellow=Entropy) to luminous RGB hex."""
    r = int(min(255, max(50, 90 + 160 * red + 110 * yellow)))
    g = int(min(255, max(60, 100 + 80 * blue + 125 * yellow)))
    b = int(min(255, max(80, 110 + 140 * blue + 35 * red)))
    return f"#{r:02x}{g:02x}{b:02x}"


def run_headless_ticks(
    state_dir: Path | str,
    ticks: int,
    idle_threshold_s: float,
    dry_run: bool,
    absorb_web: bool = False,
    absorb_dir: str = "",
    chat_test: str = "",
) -> int:
    retina = ScreenRetinaSensor()
    idle_sensor = WindowsIdleSensor()
    fg_sensor = WindowsForegroundSensor()
    # Always force dry_run=True when running non-interactively via --headless-ticks
    actuator = UniversalMotorActuator(dry_run=bool(dry_run or True))
    mind = ScaleRotorEmbodiedMind(state_dir)

    print("=" * 88)
    print("  ELYSIA — SCALE-ROTOR EMBODIED DESKTOP COMPANION (HEADLESS TICKS)")
    print("=" * 88)
    if absorb_web or absorb_dir.strip():
        dirs = [absorb_dir.strip()] if absorb_dir.strip() else []
        ab = mind.absorb_real_world_expressions(
            local_dirs=dirs,
            fetch_web=absorb_web,
            lang="ko",
        )
        print(
            f"  [현실 세계 흡수] 신규={ab['added_count']}건 | 누적현실표현={ab['total_expressions']}건 | "
            f"유도축(어근={ab.get('induced_stems', 0)}, 연산자={ab.get('induced_operators', 0)}, 스키마={ab.get('induced_schemas', 0)}) | "
            f"출처={ab['sources']}"
        )

    for i in range(max(1, int(ticks))):
        idle_s = idle_sensor.get_idle_seconds()
        fg_title = fg_sensor.get_active_window_title()
        snap = retina.capture()
        res = mind.perceive_and_act_tick(
            frame_snap=snap,
            user_idle_s=idle_s,
            agency_mode=AgencyMode.OBSERVE_AND_CHAT,
            idle_threshold_s=idle_threshold_s,
            actuator=actuator,
            foreground_window=fg_title,
        )
        c = res["chromatic"]
        f = res["feeling"]
        sc_map = res.get("universe_scales") or {}
        prec = res.get("prism_recreation") or {}
        print(
            f"  [Tick {i + 1}] 창='{res.get('foreground_window', '')[:26]}' | Action={res['action_kind']} | "
            f"Chroma(R={c['red']:.2f}, B={c['blue']:.2f}, Y={c['yellow']:.2f}) | "
            f"Feeling={f['dominant_state']}(J={f['joy']:.2f}, B={f['boredom']:.2f}, F={f['frustration']:.2f}) | "
            f"통합노드 {res.get('universe_nodes', 0)}(S0={sc_map.get(0,0)}..S4={sc_map.get(4,0)}) · 연결 {res.get('universe_links', 0)} · "
            f"도킹={res.get('docked_links', 0)} · 군집={res.get('emergent_clusters', 0)} | "
            f"프리즘(충실도={prec.get('fidelity', 0.0):.2f}, 원형={res.get('prism_prototypes', 0)}, 상관={res.get('prism_couplings', 0)}, 발견축={res.get('discovered_axes', 0)}) | "
            f"객체={len(res.get('active_objects', []))} · 접지이름={res.get('grounded_names_count', 0)} | "
            f"Knobs={res['knobs_total']} | Idle={idle_s:.1f}s"
        )

    if chat_test.strip():
        chat_res = mind.converse_with_user(
            chat_test.strip(),
            allow_autonomous_inquiry=bool(absorb_web),
        )
        rec = chat_res.get("recombination") or {}
        ddef = chat_res.get("deficit") or {}
        axes = chat_res.get("language_axes") or {}
        print(f"  [대화 시험] 사용자 : {chat_test.strip()}")
        print(f"              결핍자각: needs_inquiry={ddef.get('needs_world_inquiry')} | query='{ddef.get('inquiry_query')}' | axis={ddef.get('missing_axis')}")
        print(f"              엘리시아: {chat_res['reply_ko']}")
        print(
            f"              생성추적: grounded={rec.get('grounded_in_real_world')} | "
            f"유도 스키마={rec.get('used_schema')} | 조립 어근={rec.get('assembled_stems')} | "
            f"원문비복창={rec.get('novel_vs_corpus')} | "
            f"객체={len(chat_res.get('active_objects', []))} · 접지이름={chat_res.get('grounded_names_count', 0)} | "
            f"통합노드 {chat_res.get('universe_nodes', 0)} · 연결 {chat_res.get('universe_links', 0)} · "
            f"도킹={chat_res.get('docked_links', 0)} · 군집={chat_res.get('emergent_clusters', 0)}"
        )

    print(f"  [State Persisted] {Path(state_dir).resolve()}")
    print("=" * 88)
    return 0


class ElysiaFloatingCompanionApp:
    """Lumi-style floating desktop overlay backed by ScaleRotorEmbodiedMind."""

    def __init__(
        self,
        state_dir: Path | str,
        idle_threshold_s: float = 15.0,
        tick_ms: int = 2500,
        dry_run: bool = False,
    ) -> None:
        self.mind = ScaleRotorEmbodiedMind(state_dir)
        self.retina = ScreenRetinaSensor()
        self.idle_sensor = WindowsIdleSensor()
        self.fg_sensor = WindowsForegroundSensor()
        self.actuator = UniversalMotorActuator(dry_run=dry_run)
        self.idle_threshold_s = float(idle_threshold_s)
        self.tick_ms = max(600, int(tick_ms))

        self.vision_enabled = True
        self.agency_mode = AgencyMode.OBSERVE_AND_CHAT
        self.latest_snap: Optional[ScreenFrameSnapshot] = None
        self._anim_phase = 0.0
        self._drag_x = 0
        self._drag_y = 0

        self.root = tk.Tk()
        self.root.title("ELYSIA Companion")
        self.root.overrideredirect(True)
        self.root.attributes("-topmost", True)
        try:
            self.root.attributes("-alpha", 0.94)
        except Exception:
            pass
        self.root.configure(bg="#0d1322")

        win_w, win_h = 340, 425
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        self.root.geometry(f"{win_w}x{win_h}+{max(0, sw - win_w - 24)}+{max(0, sh - win_h - 56)}")

        # Header / drag handle
        header = tk.Frame(self.root, bg="#141e36", height=28)
        header.pack(fill=tk.X)
        title_lbl = tk.Label(
            header,
            text="ELYSIA · Scale-Rotor Companion",
            bg="#141e36",
            fg="#9ecbff",
            font=("Segoe UI", 9, "bold"),
            padx=8,
        )
        title_lbl.pack(side=tk.LEFT)
        close_btn = tk.Button(
            header,
            text="×",
            command=self._on_close,
            bg="#141e36",
            fg="#fda4af",
            bd=0,
            font=("Segoe UI", 11, "bold"),
            padx=8,
            cursor="hand2",
        )
        close_btn.pack(side=tk.RIGHT)
        for widget in (header, title_lbl):
            widget.bind("<ButtonPress-1>", self._start_drag)
            widget.bind("<B1-Motion>", self._on_drag)

        # Animated Scale-Rotor avatar canvas
        self.canvas = tk.Canvas(self.root, width=320, height=130, bg="#0d1322", highlightthickness=0)
        self.canvas.pack(pady=(6, 2))
        self.canvas.bind("<ButtonPress-1>", self._start_drag)
        self.canvas.bind("<B1-Motion>", self._on_drag)

        # Live speech bubble + learned real-world provenance
        self.bubble_var = tk.StringVar(value=self.mind.last_utterance)
        self.bubble_lbl = tk.Label(
            self.root,
            textvariable=self.bubble_var,
            bg="#16213a",
            fg="#e6edf7",
            wraplength=308,
            justify=tk.LEFT,
            font=("Segoe UI", 9),
            padx=10,
            pady=8,
            relief=tk.FLAT,
        )
        self.bubble_lbl.pack(fill=tk.X, padx=10, pady=(2, 2))

        self.provenance_var = tk.StringVar(value=self.mind.last_provenance)
        prov_lbl = tk.Label(
            self.root,
            textvariable=self.provenance_var,
            bg="#0d1322",
            fg="#94a3b8",
            wraplength=316,
            justify=tk.LEFT,
            font=("Segoe UI", 7),
            anchor="w",
        )
        prov_lbl.pack(fill=tk.X, padx=10, pady=(0, 4))

        # Status metrics & toggles
        self.status_var = tk.StringVar(value="")
        status_lbl = tk.Label(
            self.root,
            textvariable=self.status_var,
            bg="#0d1322",
            fg="#7ee787",
            font=("Segoe UI", 8),
            anchor="w",
        )
        status_lbl.pack(fill=tk.X, padx=10)

        toggles = tk.Frame(self.root, bg="#0d1322")
        toggles.pack(fill=tk.X, padx=10, pady=(4, 6))
        self.vision_btn = tk.Button(
            toggles,
            text="눈(화면 관측): ON",
            command=self._toggle_vision,
            bg="#123828",
            fg="#6ee7b7",
            bd=0,
            font=("Segoe UI", 8, "bold"),
            padx=7,
            pady=4,
            cursor="hand2",
        )
        self.vision_btn.pack(side=tk.LEFT, padx=(0, 4))

        self.roam_btn = tk.Button(
            toggles,
            text="자리비움 자율: OFF",
            command=self._toggle_roam,
            bg="#24324d",
            fg="#94a3b8",
            bd=0,
            font=("Segoe UI", 8, "bold"),
            padx=7,
            pady=4,
            cursor="hand2",
        )
        self.roam_btn.pack(side=tk.LEFT, padx=(0, 4))

        self.learn_btn = tk.Button(
            toggles,
            text="세상 배우기(문학·자막)",
            command=self._on_learn_world,
            bg="#1e293b",
            fg="#93c5fd",
            bd=0,
            font=("Segoe UI", 8, "bold"),
            padx=7,
            pady=4,
            cursor="hand2",
        )
        self.learn_btn.pack(side=tk.LEFT)

        # Chat input bar
        chat_bar = tk.Frame(self.root, bg="#0d1322")
        chat_bar.pack(fill=tk.X, padx=10, pady=(2, 10))
        self.entry = tk.Entry(
            chat_bar,
            bg="#111827",
            fg="#e6edf7",
            insertbackground="#9ecbff",
            relief=tk.FLAT,
            font=("Segoe UI", 9),
        )
        self.entry.pack(side=tk.LEFT, fill=tk.X, expand=True, ipady=5, padx=(0, 6))
        self.entry.bind("<Return>", lambda _ev: self._on_send_chat())

        send_btn = tk.Button(
            chat_bar,
            text="보내기",
            command=self._on_send_chat,
            bg="#1d4ed8",
            fg="#ffffff",
            bd=0,
            font=("Segoe UI", 9, "bold"),
            padx=10,
            pady=4,
            cursor="hand2",
        )
        send_btn.pack(side=tk.RIGHT)

        self._refresh_status_line(0.0, "observe_attend")
        self.root.after(80, self._animate_avatar)
        self.root.after(400, self._cognitive_tick)

    def _start_drag(self, event: tk.Event) -> None:
        self._drag_x = int(event.x)
        self._drag_y = int(event.y)

    def _on_drag(self, event: tk.Event) -> None:
        nx = self.root.winfo_x() + int(event.x) - self._drag_x
        ny = self.root.winfo_y() + int(event.y) - self._drag_y
        self.root.geometry(f"+{nx}+{ny}")

    def _on_close(self) -> None:
        try:
            self.mind.save()
        except Exception:
            pass
        self.root.destroy()

    def _toggle_vision(self) -> None:
        self.vision_enabled = not self.vision_enabled
        if self.vision_enabled:
            self.vision_btn.configure(text="눈(화면 관측): ON", bg="#123828", fg="#6ee7b7")
        else:
            self.vision_btn.configure(text="눈(화면 관측): OFF", bg="#24324d", fg="#94a3b8")

    def _toggle_roam(self) -> None:
        if self.agency_mode == AgencyMode.OBSERVE_AND_CHAT:
            self.agency_mode = AgencyMode.IDLE_FREE_ROAM
            self.roam_btn.configure(text="자리비움 자율: ON", bg="#3b2f13", fg="#fcd34d")
        else:
            self.agency_mode = AgencyMode.OBSERVE_AND_CHAT
            self.roam_btn.configure(text="자리비움 자율: OFF", bg="#24324d", fg="#94a3b8")

    def _on_learn_world(self) -> None:
        try:
            self.provenance_var.set("외부 문학·대화 세계에서 실제 발화를 읽는 중...")
            self.root.update_idletasks()
            res = self.mind.absorb_real_world_expressions(fetch_web=True, lang="ko")
            self.provenance_var.set(
                f"새 현실표현 {res['added_count']}건 흡수 (누적 {res['total_expressions']}건 · {', '.join(res['sources'][:2])})"
            )
            idle_s = self.idle_sensor.get_idle_seconds()
            self._refresh_status_line(idle_s, "world_absorb")
        except Exception as exc:
            self.provenance_var.set(f"세상 배우기 마찰: {exc}")

    def _refresh_status_line(self, idle_s: float, action_kind: str) -> None:
        topo = self.mind.universe.topology_summary()
        sc_map = topo["scale_counts"]
        self.status_var.set(
            f"유휴 {idle_s:.0f}s/{self.idle_threshold_s:.0f}s · {action_kind} · "
            f"노드 {topo['nodes']}(S0:{sc_map.get(0,0)}..S4:{sc_map.get(4,0)}) · "
            f"원형 {topo['prism_prototypes']} · 축 {topo['discovered_axes']} · 이름 {topo['grounded_names']}"
        )

    def _on_send_chat(self) -> None:
        text = self.entry.get().strip()
        if not text:
            return
        self.entry.delete(0, tk.END)
        try:
            res = self.mind.converse_with_user(text, frame_snap=self.latest_snap)
            self.bubble_var.set(res["reply_ko"])
            self.provenance_var.set(self.mind.last_provenance)
            idle_s = self.idle_sensor.get_idle_seconds()
            self._refresh_status_line(idle_s, "speak_recombine")
        except Exception as exc:
            self.bubble_var.set(f"대화 인지 중 마찰 발생: {exc}")

    def _animate_avatar(self) -> None:
        try:
            self._anim_phase += 0.14
            self.canvas.delete("all")
            cx, cy = 160, 66 + math.sin(self._anim_phase) * 3.5

            r, b, y = self.mind.last_chromatic
            halo_col = _chroma_to_hex(r, b, y)
            pulse = 46 + math.sin(self._anim_phase * 0.8) * 3.0

            # Outer chromatic Scale-Rotor orbital rings
            self.canvas.create_oval(
                cx - pulse - 10, cy - pulse - 10, cx + pulse + 10, cy + pulse + 10,
                outline=halo_col, width=2, dash=(4, 3),
            )
            self.canvas.create_oval(
                cx - pulse, cy - pulse, cx + pulse, cy + pulse,
                outline="#60a5fa", width=2,
            )

            # Lumi-inspired core silhouette head
            self.canvas.create_oval(cx - 28, cy - 26, cx + 28, cy + 28, fill="#1e293b", outline=halo_col, width=2)
            # Twin holographic antenna ears
            self.canvas.create_polygon(cx - 24, cy - 16, cx - 34, cy - 36, cx - 12, cy - 24, fill="#38bdf8", outline=halo_col)
            self.canvas.create_polygon(cx + 24, cy - 16, cx + 34, cy - 36, cx + 12, cy - 24, fill="#38bdf8", outline=halo_col)

            # Gaze directed toward live steered fovea coordinate (macro -> micro real screen correspondence)
            sx, sy = getattr(self.mind, "last_gaze_xy", self.mind.last_salient_xy)
            gx = (sx - 0.5) * 10.0
            gy = (sy - 0.5) * 7.0
            for ex in (cx - 10, cx + 10):
                self.canvas.create_oval(ex - 5, cy - 6, ex + 5, cy + 4, fill="#0f172a", outline="")
                self.canvas.create_oval(ex - 2.5 + gx * 0.4, cy - 3.5 + gy * 0.4, ex + 2.5 + gx * 0.4, cy + 1.5 + gy * 0.4, fill=halo_col, outline="")

            # Render persistent embodied arm/body posture synced to real screen target & OS resistance
            body_pose = getattr(self.mind, "last_body_pose", None) or {}
            if body_pose:
                sh_rad = math.radians(float(body_pose.get("shoulder_deg", 20.0)))
                el_rad = math.radians(float(body_pose.get("elbow_deg", 35.0)))
                ax0, ay0 = cx + 32, cy + 10
                ax1 = ax0 + 22 * math.cos(sh_rad - 0.6)
                ay1 = ay0 - 22 * math.sin(sh_rad - 0.6)
                ax2 = ax1 + 20 * math.cos(sh_rad + el_rad - 0.8)
                ay2 = ay1 - 20 * math.sin(sh_rad + el_rad - 0.8)
                res_val = float(body_pose.get("external_resistance", 0.0))
                arm_col = "#fb7185" if res_val >= 0.5 else "#38bdf8"
                self.canvas.create_line(ax0, ay0, ax1, ay1, ax2, ay2, fill=arm_col, width=3, smooth=True)
                self.canvas.create_oval(ax2 - 3, ay2 - 3, ax2 + 3, ay2 + 3, fill=halo_col, outline="")

            # Affective expression
            dom = self.mind.last_feeling.dominant_state if self.mind.last_feeling else "joy_expansion"
            if dom == "joy_expansion":
                self.canvas.create_arc(cx - 9, cy + 4, cx + 9, cy + 16, start=190, extent=160, style=tk.ARC, outline="#6ee7b7", width=2)
            elif dom == "friction_retreat":
                self.canvas.create_line(cx - 8, cy + 12, cx, cy + 8, cx + 8, cy + 12, fill="#fda4af", width=2)
            else:
                self.canvas.create_line(cx - 8, cy + 10, cx + 8, cy + 10, fill="#fcd34d", width=2)
        except Exception:
            pass
        self.root.after(90, self._animate_avatar)

    def _cognitive_tick(self) -> None:
        try:
            idle_s = self.idle_sensor.get_idle_seconds()
            if self.vision_enabled:
                snap = self.retina.capture()
                self.latest_snap = snap
                res = self.mind.perceive_and_act_tick(
                    frame_snap=snap,
                    user_idle_s=idle_s,
                    agency_mode=self.agency_mode,
                    idle_threshold_s=self.idle_threshold_s,
                    actuator=self.actuator,
                    foreground_window=self.fg_sensor.get_active_window_title(),
                )
                if res.get("utterance"):
                    self.bubble_var.set(str(res["utterance"]))
                    self.provenance_var.set(self.mind.last_provenance)
                self._refresh_status_line(idle_s, str(res.get("action_kind", "observe_attend")))
            else:
                self._refresh_status_line(idle_s, "vision_paused")
        except Exception as exc:
            self.bubble_var.set(f"관측 주기 마찰: {exc}")
        self.root.after(self.tick_ms, self._cognitive_tick)

    def run(self) -> int:
        self.root.mainloop()
        return 0


class ElysiaUnifiedDesktopApp:
    """Floating desktop companion powered directly by `UnifiedElysiaOrganism` (SSD<->RAM<->VRAM + GPU Rotor + 24L Rotor-Steered Qwen)."""

    def __init__(self, state_dir: Path | str, use_pretrained: bool = True) -> None:
        from core.topology.unified_elysia_organism import UnifiedElysiaOrganism

        self.org = UnifiedElysiaOrganism(
            state_dir,
            use_pretrained=use_pretrained,
            enable_llm=use_pretrained,
        )
        self.org.ingest_workspace_into_tiers(max_docs=120, max_code=120, max_images=32)
        self.org.start_living_background(capture_screen=True, spin_rotor=True)

        self.root = tk.Tk()
        self.root.title("ELYSIA — Unified Hardware Scale-Rotor Organism")
        self.root.attributes("-topmost", True)
        self.root.configure(bg="#0b1120")
        self.root.geometry("460x360+40+40")

        self.status_var = tk.StringVar(value="SSD↔RAM↔VRAM 5스케일 로터 가동 중...")
        self.bubble_var = tk.StringVar(value=self.org.last_utterance_ko)

        tk.Label(self.root, textvariable=self.status_var, fg="#38bdf8", bg="#0b1120", font=("Malgun Gothic", 9, "bold"), justify="left").pack(fill="x", padx=12, pady=(10, 4))
        tk.Label(self.root, textvariable=self.bubble_var, fg="#e2e8f0", bg="#111c33", wraplength=420, justify="left", padx=10, pady=10, font=("Malgun Gothic", 10)).pack(fill="both", expand=True, padx=12, pady=6)

        row = tk.Frame(self.root, bg="#0b1120")
        row.pack(fill="x", padx=12, pady=(4, 10))
        self.entry = tk.Entry(row, bg="#1e293b", fg="#f8fafc", insertbackground="#38bdf8", font=("Malgun Gothic", 10))
        self.entry.pack(side="left", fill="x", expand=True, padx=(0, 6))
        self.entry.bind("<Return>", lambda _e: self._on_send())
        tk.Button(row, text="대화", command=self._on_send, bg="#0284c7", fg="white", relief="flat", padx=10).pack(side="left", padx=(0, 4))
        tk.Button(row, text="창작", command=self._on_create, bg="#7c3aed", fg="white", relief="flat", padx=10).pack(side="left")

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._refresh_ui()

    def _on_send(self) -> None:
        txt = self.entry.get().strip()
        if not txt:
            return
        self.entry.delete(0, "end")
        self.bubble_var.set("로터 직결 Qwen 공명 생성 중...")
        self.root.update_idletasks()
        res = self.org.converse(txt)
        self.bubble_var.set(res["reply_ko"])

    def _on_create(self) -> None:
        self.bubble_var.set("HTML5 파동 게임 + PNG 생성 및 CLIP 자가 관측 중...")
        self.root.update_idletasks()
        art = self.org.create_and_self_observe()
        self.bubble_var.set(f"창작 완료: {Path(art['html_path']).name}\n{art['intent_note']}\n(CLIP 공명={art['visual_intent_resonance']:.3f})")

    def _refresh_ui(self) -> None:
        tel = self.org.telemetry()
        tt = tel["tiers"]
        self.status_var.set(
            f"{tel['live_hz']:.0f}Hz | VRAM {tel['vram_allocated_mb']}MB | "
            f"SSD {tt['ssd_macro_items']} ↔ RAM {tt['ram_meso_items']} ↔ VRAM {tt['vram_micro_items']} | "
            f"Hooks {tel['llm_hook_injections']} | Screen {tel['screen_frames_encoded']}"
        )
        self.root.after(800, self._refresh_ui)

    def _on_close(self) -> None:
        self.org.close()
        self.root.destroy()

    def run(self) -> int:
        self.root.mainloop()
        return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="ELYSIA Floating Scale-Rotor Desktop Companion")
    parser.add_argument("--organism", action="store_true", help="Launch Unified Organism (SSD<->RAM<->VRAM + 5-Scale GPU Rotor + 24L Rotor-Steered Qwen)")
    parser.add_argument(
        "--state-dir",
        type=str,
        default=r"c:\Elysia\data\elysia_embodied_state",
        help="Persistent directory storing Scale-Rotor embodied state",
    )
    parser.add_argument(
        "--idle-threshold",
        type=float,
        default=15.0,
        help="User inactivity seconds required before idle free-roam actions activate",
    )
    parser.add_argument(
        "--tick-ms",
        type=int,
        default=2500,
        help="Cognitive perception tick interval in milliseconds",
    )
    parser.add_argument(
        "--headless-ticks",
        type=int,
        default=0,
        help="Run N non-GUI headless perception ticks (dry-run) and exit cleanly",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log physical OS/browser actions without moving real mouse/keyboard or opening windows",
    )
    parser.add_argument(
        "--absorb-web",
        action="store_true",
        help="Fetch public Korean Wikisource / Gutenberg literature to learn real situated human expressions",
    )
    parser.add_argument(
        "--absorb-dir",
        type=str,
        default="",
        help="Local directory containing .srt/.vtt subtitles, books, and paired .wav/.mp4 files to learn",
    )
    parser.add_argument(
        "--chat-test",
        type=str,
        default="",
        help="Optional Korean utterance to test real-world situated dialogue response in headless mode",
    )
    args = parser.parse_args(argv)

    if args.organism:
        return ElysiaUnifiedDesktopApp(state_dir=args.state_dir).run()

    if args.headless_ticks > 0:
        return run_headless_ticks(
            state_dir=args.state_dir,
            ticks=args.headless_ticks,
            idle_threshold_s=args.idle_threshold,
            dry_run=args.dry_run,
            absorb_web=args.absorb_web,
            absorb_dir=args.absorb_dir,
            chat_test=args.chat_test,
        )

    app = ElysiaFloatingCompanionApp(
        state_dir=args.state_dir,
        idle_threshold_s=args.idle_threshold,
        tick_ms=args.tick_ms,
        dry_run=args.dry_run,
    )
    return app.run()


if __name__ == "__main__":
    raise SystemExit(main())
