r"""
ELYSIA Autonomous Homoiconic Protocol Fabric Runner
===================================================

ZERO `--interactive`, ZERO `input()`, ZERO hardcoded labels or channel menus.
Elysia discovers real protocols actually present on THIS OS, encounters uncurated
live OS & workspace phenomena, lets data/protocols/lenses/causalities/orders self-weave
via ONE Binding Principle, and autonomously weaves (and optionally enacts via `--enact-contact`)
an external OS protocol chain to `human_surface` whenever unresolved tension arises.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
L1_ROOT = ROOT / "L1_causal_field"
L2_ROOT = ROOT / "L2_cognitive_engine"
for p in (str(ROOT), str(L1_ROOT), str(L2_ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

from core.embodiment.os_protocol_discovery import (
    capture_live_environment_seed,
    discover_real_os_protocols,
)
from core.topology.protocol_weaving_fabric import BindingField


def main() -> None:
    parser = argparse.ArgumentParser(description="ELYSIA Autonomous Homoiconic Protocol Fabric")
    parser.add_argument(
        "--enact-contact",
        action="store_true",
        help="Allow Elysia to actually execute her self-woven OS human-contact protocol on the desktop",
    )
    parser.add_argument("--steps", type=int, default=4, help="Autonomous self-weaving cycles to run")
    parser.add_argument("--files", type=int, default=6, help="Number of uncurated workspace files to expose")
    args = parser.parse_args()

    field = BindingField()
    protocols = discover_real_os_protocols(ROOT)
    for proto in protocols:
        field.introduce(proto, why_ko=f"실제 OS 런타임 탐색으로 발견된 프로토콜 `{proto.structure_id}`")

    env_seeds = capture_live_environment_seed(ROOT, file_limit=max(1, args.files))
    for seed in env_seeds:
        field.introduce(seed, why_ko=f"무라벨 외부 현실 접점 `{seed.structure_id}` 유입")

    print(
        f"[OS Discovery] 실제 OS에서 발견한 프로토콜 {len(protocols)}개 · "
        f"무라벨 환경 접점 {len(env_seeds)}개 유입 (CLI input 대기 없음)"
    )

    for _ in range(max(1, args.steps)):
        res = field.step_self_weave(allow_outward_enact=args.enact_contact)
        ev = res["events"]
        print(
            f"  [step {res['step']:02d}] 프로토콜확장=+{len(ev['extended_protocols'])} "
            f"| 실세계접합={len(ev['enacted'])} | 동형굴절={len(ev['refracted'])} "
            f"| 외부인간경로직조={len(ev['outward_chains'])}"
        )

    obs = field.observe_fabric()
    print()
    print(obs.render_ko())
    saved = field.save_trace(ROOT / "data" / "protocol_weaving_fabric_trace.json")
    print(f"\n[Saved] {saved}")


if __name__ == "__main__":
    main()
