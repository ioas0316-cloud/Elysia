# [Genesis: 2025-12-02] Purified by Elysia
"""
Elysia avatar agent (META layer, 2025-11-16).

Purpose
- Choose a small set of WORLD characters to act as "avatars" or
  화신 for Elysia, based on the current world snapshot.
- This does not change WORLD physics; it only emits metadata that
  Godot / higher layers can use to highlight these agents and reason
  about their roles.

Outputs
- logs/elysia_avatars.json
  {
    "avatars": [
      {
        "id": "citizen_3",
        "name": "...",
        "role_hint": "literate_hero",
        "alignment_tag": "lawful_good",
        "power_score": 72.5,
        "sign_text_ko": "...",
        "sign_text_en": "...",
        "diary_id": "diary_citizen_3"
      },
      ...
    ]
  }
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional


def _ensure_repo_root_on_path() -> str:
    here = os.path.abspath(os.path.dirname(__file__))
    repo_root = os.path.abspath(os.path.join(here, ".."))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    return repo_root


_ensure_repo_root_on_path()


def _load_world_snapshot(logs_dir: str) -> Dict[str, Any]:
    path = os.path.join(logs_dir, "world_snapshot_for_godot.json")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _index_text_objects_by_owner(snapshot: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    text_objects = snapshot.get("text_objects", []) or []
    by_owner: Dict[str, List[Dict[str, Any]]] = {}
    for obj in text_objects:
        owner_id = obj.get("owner_id")
        if not owner_id:
            continue
        by_owner.setdefault(owner_id, []).append(obj)
    return by_owner


def _choose_role_hint(
    alignment_tag: str,
    power_score: float,
    has_diary: bool,
    has_sign: bool,
) -> str:
    if has_diary and alignment_tag in ("lawful_good", "good"):
        return "literate_hero"
    if has_diary and alignment_tag in ("chaotic_good", "neutral"):
        return "wandering_scholar"
    if alignment_tag in ("chaotic_evil", "lawful_evil"):
        return "villain_candidate"
    if power_score > 80.0:
        return "high_master"
    if has_sign:
        return "literate_commoner"
    return "ordinary"


def build_avatars_from_snapshot(
    snapshot: Dict[str, Any],
    max_avatars: int = 8,
) -> List[Dict[str, Any]]:
    chars = snapshot.get("characters", []) or []
    text_by_owner = _index_text_objects_by_owner(snapshot)

    # Filter potential candidates: those with diaries or signs, or high power.
    candidates: List[Dict[str, Any]] = []
    for ch in chars:
        cid = ch.get("id")
        if not cid:
            continue
        power = float(ch.get("power_score", 0.0))
        align_tag = str(ch.get("alignment_tag") or "neutral")
        sign_ko = ch.get("sign_text_ko")
        sign_en = ch.get("sign_text_en")
        has_sign = bool(sign_ko or sign_en)
        diaries = [
            obj
            for obj in text_by_owner.get(cid, [])
            if obj.get("kind") == "diary"
        ]
        has_diary = bool(diaries)

        # Soft filter: either literate, diary owner, or quite powerful.
        if not (has_diary or has_sign or power > 60.0):
            continue

        role_hint = _choose_role_hint(align_tag, power, has_diary, has_sign)
        diary_id: Optional[str] = diaries[0].get("id") if diaries else None

        candidates.append(
            {
                "id": cid,
                "name": ch.get("name"),
                "alignment_tag": align_tag,
                "power_score": power,
                "sign_text_ko": sign_ko,
                "sign_text_en": sign_en,
                "diary_id": diary_id,
                "role_hint": role_hint,
            }
        )

    # Sort by a simple priority: diaries first, then power.
    def _score(c: Dict[str, Any]) -> float:
        has_diary = 1.0 if c.get("diary_id") else 0.0
        return has_diary * 1000.0 + float(c.get("power_score", 0.0))

    candidates.sort(key=_score, reverse=True)
    return candidates[:max_avatars]


def main() -> None:
    repo_root = _ensure_repo_root_on_path()
    logs_dir = os.path.join(repo_root, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    snapshot = _load_world_snapshot(logs_dir)
    avatars = build_avatars_from_snapshot(snapshot)

    out_path = os.path.join(logs_dir, "elysia_avatars.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"avatars": avatars}, f, ensure_ascii=False, indent=2)

    print(f"[elysia_avatar_agent] Wrote {len(avatars)} avatars to: {out_path}")


if __name__ == "__main__":
    main()
