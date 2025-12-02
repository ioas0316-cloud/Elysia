# [Genesis: 2025-12-02] Purified by Elysia
#!/usr/bin/env python3
"""Generate Codex §24-compliant experiment report templates.

Caretakers on low-spec hardware can run this script to pre-fill the
required JSON skeleton before adding actual metrics/log references.
"""
from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from typing import List


def _comma_list(value: str | None) -> List[str]:
    if not value:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def build_template(args: argparse.Namespace) -> dict:
    now = _dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    return {
        "ts": now,
        "trial_id": args.trial_id,
        "branch_plan_id": args.branch_plan_id,
        "world_kit": args.world_kit,
        "body_architecture": args.body_architecture,
        "observer_architecture": args.observer_architecture,
        "level_id": args.level_id,
        "macro_years": args.macro_years,
        "time_scale": args.time_scale,
        "seeds": {
            "total": args.seeds_total,
            "completed": args.seeds_completed,
        },
        "language_axes": _comma_list(args.language_axes),
        "plan_status": args.plan_status,
        "status": args.status,
        "status_history": [
            {
                "ts": now,
                "status": args.status,
                "actor": args.actor,
                "notes": args.notes,
            }
        ],
        "execution_evidence": {
            "macro_ticks_completed": args.macro_ticks_completed,
            "seeds_completed": args.seeds_completed,
            "self_writing_samples": args.self_writing_samples,
            "resonance_avg": args.resonance_avg,
            "language_field_delta": args.language_field_delta,
            "evidence_refs": _comma_list(args.evidence_refs),
        },
        "blocking_reason": args.blocking_reason,
        "purpose": args.purpose,
        "method": args.method,
        "observations": args.observations,
        "integration": args.integration,
        "references": _comma_list(args.references),
        "adult_ready": args.adult_ready,
        "adult_readiness_notes": args.adult_readiness_notes,
        "summary": args.summary,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create a JSON skeleton for macro-scale trials that satisfies "
            "the Experiment Design Guide and Codex §24 reporting rules."
        )
    )
    parser.add_argument("--trial-id", required=True)
    parser.add_argument("--branch-plan-id", required=True)
    parser.add_argument("--world-kit", required=True)
    parser.add_argument("--body-architecture", required=True)
    parser.add_argument("--observer-architecture", default="")
    parser.add_argument("--level-id", required=True)
    parser.add_argument("--macro-years", type=int, default=1000)
    parser.add_argument("--time-scale", default="1 tick = 1 week")
    parser.add_argument("--seeds-total", type=int, default=20)
    parser.add_argument("--seeds-completed", type=int, default=0)
    parser.add_argument("--language-axes", default="")
    parser.add_argument("--plan-status", default="planning")
    parser.add_argument("--status", default="awaiting_execution")
    parser.add_argument("--actor", default="caretaker")
    parser.add_argument("--notes", default="template generated")
    parser.add_argument("--macro-ticks-completed", type=int, default=0)
    parser.add_argument("--self-writing-samples", type=int, default=0)
    parser.add_argument("--resonance-avg", type=float, default=0.0)
    parser.add_argument("--language-field-delta", type=float, default=0.0)
    parser.add_argument("--evidence-refs", default="")
    parser.add_argument("--blocking-reason", default="")
    parser.add_argument("--purpose", default="")
    parser.add_argument("--method", default="")
    parser.add_argument("--observations", default="")
    parser.add_argument("--integration", default="")
    parser.add_argument("--references", default="")
    parser.add_argument("--adult-ready", action="store_true")
    parser.add_argument("--adult-readiness-notes", default="")
    parser.add_argument("--summary", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    template = build_template(args)
    json.dump(template, fp=sys.stdout, ensure_ascii=False, indent=2)
    print()


if __name__ == "__main__":
    main()