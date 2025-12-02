# [Genesis: 2025-12-02] Purified by Elysia
"""
Ultra‑simple console chat for Elysia without a web server.

Features
- Type to chat; enter to send
- Commands:
  - /sleep        → run one dream cycle (Cellular World simulation)
  - /recent       → show recent emergent meaning:* (from CoreMemory)
  - /help         → show help
  - /quit         → exit

Notes
- Runs CognitionPipeline directly. A lightweight Cellular World is
  created in‑process so you can see emergence without Flask.
"""
from __future__ import annotations

# Ensure project root on sys.path before any local imports
import sys as _sys
from pathlib import Path as _Path
_ROOT = _Path(__file__).resolve().parent.parent
if str(_ROOT) not in _sys.path:
    _sys.path.insert(0, str(_ROOT))

import os
import sys
from typing import List

from Project_Elysia.cognition_pipeline import CognitionPipeline
import re
import json
import os
from Project_Sophia.core.world import World
from tools.kg_manager import KGManager
from Project_Elysia.core_memory import CoreMemory


PRIMORDIAL_DNA = {
    "instinct": "connect_create_meaning",
    "resonance_standard": "love",
}


def _soul_sync(world: World, kg: dict) -> None:
    """Mirror KG → Cellular World (nodes + simple edges)"""
    # nodes
    for node in kg.get("nodes", []):
        nid = node.get("id")
        if not nid:
            continue
        if nid not in world.cells:
            energy = float(node.get("activation_energy", 0.0) or 0.0)
            world.add_cell(nid, properties=node, initial_energy=energy)
    # edges (directional)
    for e in kg.get("edges", []):
        s, t = e.get("source"), e.get("target")
        if not s or not t:
            continue
        cs, ct = world.get_cell(s), world.get_cell(t)
        if not cs or not ct or not cs.is_alive or not ct.is_alive:
            continue
        if not any(c.get("target_id") == t for c in cs.connections):
            strength = float(e.get("strength", 0.5) or 0.5)
            cs.connect(ct, relationship_type=e.get("relation", "related_to"), strength=strength)


def _plain_to_ids(msg: str, kg: dict) -> list[str]:
    """Map plain Korean words to KG node ids by suffix match (:단어)."""
    ids = [n.get('id') for n in kg.get('nodes', []) if n.get('id')]
    words = list(set(re.findall(r"[가-힣]{1,20}", msg or '')))

    # Normalize simple Korean particles/endings to base nouns (rough heuristic)
    def norm(tok: str) -> str:
        if not tok:
            return tok
        # strip common 조사가/격조사/보조사
        suffixes = [
            '으로', '에서', '에게', '까지', '부터', '이라', '라서',
            '은','는','이','가','을','를','와','과','의','에','로','도','만'
        ]
        changed = True
        while changed and len(tok) > 1:
            changed = False
            for s in suffixes:
                if tok.endswith(s) and len(tok) > len(s):
                    tok = tok[:-len(s)]
                    changed = True
                    break
        # simple verb ending: 사랑해 → 사랑
        for v in ['했다','해요','하네','하는','하다','해','했어','했니']:
            if tok.endswith(v) and len(tok) > len(v):
                tok = tok[:-len(v)]
                break
        return tok

    normed = set()
    for w in words:
        nw = norm(w)
        if nw:
            normed.add(nw)
    result = []
    # Load aliases (plain -> canonical) if available
    aliases_path = os.path.join('data', 'lexicon', 'aliases_ko.json')
    aliases = {}
    try:
        with open(aliases_path, 'r', encoding='utf-8') as f:
            aliases = json.load(f)
    except Exception:
        aliases = {}

    # Special disambiguation for '말' (language vs animal). Default to language.
    if '말' in normed:
        # Prefer explicit '언어' node if present
        if any(nid.endswith(':언어') for nid in ids):
            result.append(next(nid for nid in ids if nid.endswith(':언어')))
        # Also include an explicit ':말' node if it exists (for users who mean the animal)
        for nid in ids:
            if nid.endswith(':말'):
                result.append(nid)
        # Remove the raw '말' from further generic matching to avoid duplicates
        normed.discard('말')

    # Apply aliases
    for w in list(normed):
        if w in aliases:
            normed.add(aliases[w])
    for w in normed:
        for nid in ids:
            if nid.endswith(':' + w):
                result.append(nid)
    return list(set(result))


def _teach_alias(raw: str) -> str:
    """Parse 'teach: A=B' and update aliases file."""
    m = re.match(r"^\s*teach\s*:\s*(.+?)\s*[=\-\>]\s*(.+)\s*$", raw, re.IGNORECASE)
    if not m:
        return "형식: teach: A=B (예: teach: 말=언어)"
    a, b = m.group(1).strip(), m.group(2).strip()
    path = os.path.join('data', 'lexicon', 'aliases_ko.json')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {}
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        data = {}
    data[a] = b
    try:
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        return f"학습 완료: {a} → {b}"
    except Exception as e:
        return f"저장 실패: {e}"


def _dream_once(world: World, cm: CoreMemory) -> List[str]:
    """Run one simulation step and return new meaning ids (best‑effort)."""
    before = set(world.cells.keys())
    newborn = world.run_simulation_step()
    after = set(world.cells.keys())
    new_ids = [cid for cid in (after - before) if cid.startswith("meaning:")]
    # Best‑effort: also surface notable hypotheses if any
    hyps = cm.data.get("notable_hypotheses", [])
    if new_ids and hyps:
        pass
    return new_ids


def main() -> None:
    print("Elysia Console Chat — no web server needed. Type /help for commands.\n")
    # Initialize components
    kgm = KGManager()
    world = World(primordial_dna=PRIMORDIAL_DNA)
    cm = CoreMemory()
    pipe = CognitionPipeline(cellular_world=world)
    # Initial sync
    _soul_sync(world, kgm.kg)

    def _help():
        print("\nCommands:\n  /sleep   → run a dream cycle (simulate emergence)\n  /recent  → list recent emergent meaning:*\n  /help    → show this help\n  /quit    → exit\n")

    _help()
    while True:
        try:
            msg = input("You> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
        if not msg:
            continue
        if msg == "/quit":
            print("Bye.")
            break
        if msg == "/help":
            _help()
            continue
        if msg.lower().startswith('teach:'):
            print(_teach_alias(msg))
            # refresh aliases by re-mirroring on next use
            continue
        if msg == "/recent":
            hyps = cm.data.get("notable_hypotheses", [])
            if not hyps:
                print("(no recent emergent meanings yet)")
            else:
                print("Recent meaning:*")
                for h in hyps[-5:][::-1]:
                    head, tail = h.get("head", "?"), h.get("tail", "?")
                    conf = h.get("confidence")
                    name = f"meaning:{head}_{tail}"
                    if conf is None:
                        print(f" - {name}")
                    else:
                        print(f" - {name} (conf={conf})")
            continue
        if msg == "/sleep":
            # Keep KG in sync, then dream once
            _soul_sync(world, kgm.kg)
            new_ids = _dream_once(world, cm)
            if new_ids:
                print("Emergence:")
                for nid in new_ids:
                    print(f" - {nid}")
            else:
                print("(no new meaning this cycle)")
            continue

        # Regular chat
        try:
            # Nutrients for plain words
            ids = _plain_to_ids(msg, kgm.kg)
            if ids:
                fed = []
                for nid in ids:
                    c = world.get_cell(nid)
                    if not c:
                        # ensure cell exists when mirrored
                        if kgm.get_node(nid):
                            world.add_cell(nid, properties=kgm.get_node(nid), initial_energy=0.0)
                            c = world.get_cell(nid)
                    if c and c.is_alive:
                        c.add_energy(1.0)
                        fed.append(nid.split(':',1)[-1])
                if fed:
                    print("(에너지를 먹였어요):", ", ".join(fed))
            resp, _state = pipe.process_message(msg)
            if isinstance(resp, dict) and resp.get("type") == "text":
                print("Elysia>", resp.get("text", ""))
            else:
                print("Elysia>", str(resp))
        except Exception as e:
            print("(error)", e)


if __name__ == "__main__":
    # Ensure project root on sys.path when run as module
    sys.exit(main())