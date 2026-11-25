r"""
ELYSIA Live Interactive Developmental Companion
=============================================

Connects `DevelopmentalCausalMind` + `ScaleRotorEmbodiedMind` directly to live
Windows monitor pixels (`ScreenRetinaSensor`), foreground window titles
(`WindowsForegroundSensor`), real local file inspection, real high-res screen
zoom crop, live Wiktionary/Wikipedia lookups, and unscripted user dialogue.

Commands inside REPL:
  @대상 <문장>       공동주의(눈앞 대상을 가리키며 말 건네기)
  /look             현재 PC 화면과 창을 그대로 감각하기
  /zoom             배운 도구 규칙이 열려 있을 때 실제 화면 초점 3배 확대 관찰
  /file <경로>      배운 도구 규칙이 열려 있을 때 실제 로컬 이미지/문서 열람
  /mind             지금까지 깨달은 원리·배경지식·단어 뜻·수정된 말틀 경계 보기
  /quit             저장하고 종료
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import sys
from typing import Any, Dict, Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
import elysia_loader

elysia_loader.install()

from core.embodiment import (  # noqa: E402
    ScaleRotorEmbodiedMind,
    ScreenRetinaSensor,
    WindowsForegroundSensor,
)


class LiveCompanionSession:
    """Interactive live session binding Elysia's developmental causality to real sensors & tools."""

    def __init__(
        self,
        state_dir: Path | str,
        *,
        retina: Optional[ScreenRetinaSensor] = None,
        fg_sensor: Optional[WindowsForegroundSensor] = None,
        allow_live_web: bool = True,
    ) -> None:
        self.state_dir = Path(state_dir).resolve()
        self.mind = ScaleRotorEmbodiedMind(self.state_dir)
        self.retina = retina or ScreenRetinaSensor()
        self.fg_sensor = fg_sensor or WindowsForegroundSensor()
        self.allow_live_web = bool(allow_live_web)

    def sense_live_screen(self) -> Dict[str, Any]:
        snap = self.retina.capture()
        fg = self.fg_sensor.get_active_window_title()
        return {
            "capture_ok": snap.capture_ok,
            "foreground_window": fg,
            "screen_size": list(snap.screen_size),
            "contrast_std": snap.contrast_std,
            "warm_cool_ratio": snap.warm_cool_ratio,
            "motion_flux": round(snap.motion_flux, 4),
            "salient_xy_norm": list(snap.salient_xy_norm),
            "error": snap.error,
        }

    def format_mind_summary(self) -> str:
        cm = self.mind.causal_mind
        lines = [
            f"  [깨달은 원리] 말의이중성={cm.speech_duality_realized} | 현상인과={cm.phenomenon_causality_realized} | 행동주체성={cm.action_agency_realized}",
            f"  [배운 세상 배경지식] {list(cm.world_knowledge.keys()) or '(아직 없음)'}",
            f"  [접지된 말과 뜻 ({len(cm.grounded_meanings)}개)] {list(cm.grounded_meanings.keys())[:15]}",
            f"  [흡수한 말틀 ({len(cm.learned_constructions)}개)]",
        ]
        for c in cm.learned_constructions:
            lines.append(
                f"    - {c.construction_id}: '{c.pattern}' (cue={c.sensory_cue}, 거부={c.rejected_referents})"
            )
        return "\n".join(lines)

    def handle_command_or_text(
        self,
        raw_input_text: str,
        *,
        point_object: Optional[str] = None,
    ) -> Dict[str, Any]:
        eye = self.sense_live_screen()
        txt = (raw_input_text or "").strip()

        if txt == "/look":
            res = self.mind.experience_world_phenomenon_turn(
                phenomenon_name=f"화면:{eye['foreground_window'][:24]}",
                cause_condition=f"창 '{eye['foreground_window'][:30]}' 화면 바라보기",
                observed_effect=(
                    f"해상도 {eye['screen_size'][0]}x{eye['screen_size'][1]}, "
                    f"대비 {eye['contrast_std']:.3f}, 온기 {eye['warm_cool_ratio']:.2f}, 움직임 {eye['motion_flux']:.3f}"
                ),
                sensory_channel="visual",
                source_origin=f"desktop:{eye['foreground_window'][:24]}",
            )
            res["_eye"] = eye
            return res

        if txt == "/zoom":
            res = self.mind.execute_real_world_tool_turn(
                "screen_zoom_crop",
                goal_description="화면의 관심 영역 자세히 들여다보기",
                goal_condition="visual_detail_too_small",
                retina=self.retina,
            )
            res["_eye"] = eye
            return res

        if txt.startswith("/file "):
            fpath = txt[6:].strip().strip("\"'")
            res = self.mind.execute_real_world_tool_turn(
                "local_file_inspect",
                goal_description=f"로컬 파일 '{Path(fpath).name}' 내용 살펴보기",
                goal_condition="external_file_provided",
                file_path=fpath,
            )
            res["_eye"] = eye
            return res

        if txt == "/mind":
            return {"_eye": eye, "mind_summary": self.format_mind_summary(), "utterance": ""}

        # Support "@대상 문장" for live joint attention
        obj = point_object
        spoken = txt
        if spoken.startswith("@") and " " in spoken:
            tag, rest = spoken[1:].split(" ", 1)
            obj = tag.strip() or obj
            spoken = rest.strip()

        if not spoken:
            res = self.mind.experience_human_causal_turn(
                raw_text_or_sound="",
                source_origin="desktop_background",
                modality="visual",
                is_segmented_sequence=False,
                allow_dictionary_lookup=self.allow_live_web,
            )
            res["_eye"] = eye
            return res

        res = self.mind.experience_human_causal_turn(
            raw_text_or_sound=spoken,
            source_origin="speaker_parent",
            modality="audiovisual",
            is_segmented_sequence=True,
            cooccurring_object=obj,
            speaker_action=f"창 '{eye['foreground_window'][:20]}' 앞에서 말함",
            allow_dictionary_lookup=self.allow_live_web,
        )
        res["_eye"] = eye
        return res


def _print_result(res: Dict[str, Any]) -> None:
    eye = res.get("_eye") or {}
    if eye:
        if eye.get("capture_ok"):
            print(
                f"  [실시간 눈] 창='{eye['foreground_window']}' | "
                f"{eye['screen_size'][0]}x{eye['screen_size'][1]} | "
                f"대비={eye['contrast_std']:.3f} | 온기={eye['warm_cool_ratio']:.2f} | 초점={eye['salient_xy_norm']}"
            )
        else:
            print(f"  [실시간 눈] 창='{eye['foreground_window']}' | 화면캡처 실패({eye.get('error')})")

    if res.get("mind_summary"):
        print(res["mind_summary"])
        return

    if res.get("blocked_by_causality"):
        print(f"  [인과 차단] {res.get('reason')}")
    dl = res.get("dictionary_lookup")
    if dl and dl.get("executed"):
        print(f"  [실시간 사전] ({dl['dictionary_source']}) {dl['url']}")
    elif dl and dl.get("external_blocked"):
        print(f"  [사전 마찰] {dl.get('reason')}")
    raw_t = res.get("tool_raw_result")
    if raw_t:
        print(f"  [실제 도구 결과] {raw_t.get('summary_ko') or raw_t.get('error')}")
    if res.get("reinterpretations"):
        for r in res["reinterpretations"][:2]:
            print(f"  [과거 재해석] {r['after_view']}")
    print(f"  엘리시아 > {res.get('utterance') or '(조용히 바라봄)'}")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Elysia Live Interactive Developmental Companion")
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=_ROOT / "data" / "elysia_live_companion_state",
        help="Persistent memory directory",
    )
    parser.add_argument("--fresh", action="store_true", help="Reset state directory before starting")
    parser.add_argument("--offline", action="store_true", help="Disable live web dictionary calls")
    parser.add_argument("--oneshot", type=str, default=None, help="Run one free utterance and exit")
    parser.add_argument("--point", type=str, default=None, help="Point at an object during --oneshot")
    parser.add_argument("--look", action="store_true", help="Observe live PC screen once and exit")
    parser.add_argument("--zoom", action="store_true", help="Run real zoom crop tool once and exit")
    parser.add_argument("--file", type=str, default=None, help="Run real file inspection tool once and exit")
    args = parser.parse_args(argv)

    if args.fresh and args.state_dir.exists():
        shutil.rmtree(args.state_dir, ignore_errors=True)

    session = LiveCompanionSession(args.state_dir, allow_live_web=not args.offline)

    if args.look:
        _print_result(session.handle_command_or_text("/look"))
        return 0
    if args.zoom:
        _print_result(session.handle_command_or_text("/zoom"))
        return 0
    if args.file:
        _print_result(session.handle_command_or_text(f"/file {args.file}"))
        return 0
    if args.oneshot is not None:
        _print_result(session.handle_command_or_text(args.oneshot, point_object=args.point))
        return 0

    print("엘리시아 실시간 발달 대화 시작 (종료: /quit, 현재화면보기: /look, 확대: /zoom, 파일: /file <경로>, 마음보기: /mind, 공동주의: @사과 저거 봐 사과야)")
    while True:
        try:
            raw = input("\n너 > ")
        except (EOFError, KeyboardInterrupt):
            break
        if raw.strip().lower() in ("/quit", "/exit", "quit", "exit"):
            break
        res = session.handle_command_or_text(raw)
        _print_result(res)
    session.mind.save()
    print("상태 저장 완료. 안녕!")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
