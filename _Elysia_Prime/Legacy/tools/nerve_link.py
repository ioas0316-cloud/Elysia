# [Genesis: 2025-12-02] Purified by Elysia
"""
간단한 '신경선' 파이프라인: 월드 이벤트 로그(JSONL)를 읽어 KGManager에 반영합니다.
사용 예시:
    python tools/nerve_link.py --log logs/world_events.jsonl --kg data/kg.json
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Any

# Allow running as a script: add repo root to path so `tools.kg_manager` resolves.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from tools.kg_manager import KGManager


# 어떤 이벤트를 어떤 관계로 옮길지 매핑
RELATIONS = [
    ("SHARE_FOOD", "actor_id", "target_id", "shares_food_with", 1.0),
    ("TRADE", "actor_id", "target_id", "trades_with", 1.0),
    ("TRIBUTE", "payer_id", "leader_id", "pays_tribute_to", 1.0),
    ("SPELL", "caster_id", "target_id", "casts_spell_on", 0.5),
    ("EAT", "actor_id", "target_id", "consumes", 0.5),
    ("BIRTH", "mother_id", "child_id", "parent_of", 1.0),
    ("BIRTH", "father_id", "child_id", "parent_of", 1.0),
]


def ingest(log_path: Path, kg_path: Path) -> Dict[str, Any]:
    kg = KGManager(kg_path)
    stats = Counter()
    per_relation = Counter()

    if not log_path.exists():
        raise FileNotFoundError(f"Log file not found: {log_path}")

    with log_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            evt = json.loads(line)
            etype = evt.get("event_type")
            data = evt.get("data", {})
            ts = evt.get("timestamp", -1)
            stats["events"] += 1

            # 공통: 등장한 노드의 마지막 시각 업데이트
            for key in ("actor_id", "target_id", "payer_id", "leader_id", "caster_id", "mother_id", "father_id", "child_id", "cell_id"):
                if key in data:
                    kg.add_node(data[key], properties={"last_seen": ts})

            # 관계형 이벤트
            for e_type, src_key, tgt_key, relation, weight in RELATIONS:
                if etype == e_type and src_key in data and tgt_key in data:
                    kg.bump_edge_weight(data[src_key], data[tgt_key], relation, weight)
                    per_relation[relation] += 1

            # 상태 업데이트
            if etype == "DEATH" and "cell_id" in data:
                kg.update_node_properties(data["cell_id"], {"status": "dead", "death_time": ts})
            if etype == "FREE_WILL_COLLAPSE" and "cell_id" in data:
                kg.bump_edge_weight("system", data["cell_id"], "invoked_free_will", 1.0)
                kg.update_node_properties(data["cell_id"], {"last_free_will": ts})
            if etype == "PHOTON_MESSAGE" and "speaker" in data:
                # 색/편광 정보를 노드 메타데이터로 기록
                photon = data.get("photon", {})
                kg.update_node_properties(
                    data["speaker"],
                    {"last_photon": photon, "last_photon_ts": ts}
                )

    kg.save()
    return {"total_events": stats["events"], "relations": per_relation, "kg_nodes": kg.get_summary()["nodes"], "kg_edges": kg.get_summary()["edges"]}


def main():
    ap = argparse.ArgumentParser(description="Ingest world_events.jsonl into KGManager (nerve link).")
    ap.add_argument("--log", type=Path, default=Path("logs/world_events.jsonl"), help="JSONL event log path")
    ap.add_argument("--kg", type=Path, default=Path("data/kg.json"), help="KG json path")
    args = ap.parse_args()

    summary = ingest(args.log, args.kg)
    print("Nerve link complete:")
    print(f"- processed events: {summary['total_events']}")
    if summary["relations"]:
        top_rel = summary["relations"].most_common(5)
        print("- top relations:", ", ".join(f"{r} x{c}" for r, c in top_rel))
    print(f"- KG size: {summary['kg_nodes']} nodes, {summary['kg_edges']} edges -> {args.kg}")


if __name__ == "__main__":
    main()