import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"


def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _summarize_wave_state(state: Dict[str, Any]) -> Dict[str, Any]:
    alerts = state.get("alerts", [])
    return {
        "total_waves": state.get("total_waves"),
        "organs": {k: v.get("health") for k, v in state.get("organs", {}).items()},
        "alerts": alerts,
        "alert_count": len(alerts),
    }


def _summarize_immune_state(state: Dict[str, Any]) -> Dict[str, Any]:
    stats = state.get("stats", {})
    return {
        "gates": state.get("ozone", {}).get("gates"),
        "blocked_rate": state.get("ozone", {}).get("block_rate"),
        "stats": {
            "blocked": stats.get("threats_blocked"),
            "neutralized": stats.get("threats_neutralized"),
            "cells_deployed": stats.get("cells_deployed"),
            "signals": stats.get("signals_transmitted"),
        },
        "dna_self_signature": state.get("dna_self_signature"),
    }


def _summarize_nanocells(report: Dict[str, Any]) -> Dict[str, Any]:
    cells = report.get("cells", [])
    found = sum(c.get("issues_found", 0) for c in cells)
    fixed = sum(c.get("issues_fixed", 0) for c in cells)
    return {"issues_found": found, "issues_fixed": fixed, "cells": len(cells)}


def _summarize_evaluation(latest: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "score": latest.get("total_score"),
        "max": latest.get("max_score"),
        "grade": latest.get("grade"),
        "communication_pct": latest.get("communication", {}).get("percentage"),
        "thinking_pct": latest.get("thinking", {}).get("percentage"),
        "timestamp": latest.get("evaluation_time"),
    }


def build_snapshot() -> Dict[str, Any]:
    wave_state = _load_json(DATA_DIR / "wave_organization_state.json")
    immune_state = _load_json(DATA_DIR / "immune_system_state.json")
    nanocell_report = _load_json(DATA_DIR / "nanocell_report.json")
    latest_eval = _load_json(DATA_DIR.parent / "reports" / "evaluation_latest.json")
    registry = _load_json(DATA_DIR / "central_registry.json")

    snapshot = {
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "wave_state": _summarize_wave_state(wave_state),
        "immune_state": _summarize_immune_state(immune_state),
        "nanocells": _summarize_nanocells(nanocell_report),
        "latest_evaluation": _summarize_evaluation(latest_eval),
    }

    # Keep the current registry and attach runtime state.
    registry["runtime_state"] = snapshot
    _write_json(DATA_DIR / "central_registry.json", registry)
    _write_json(DATA_DIR / "system_status_snapshot.json", snapshot)
    return snapshot


if __name__ == "__main__":
    summary = build_snapshot()
    print("Runtime snapshot stored.")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
