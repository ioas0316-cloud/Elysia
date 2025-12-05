"""
Elysia Auto‑Pilot

Runs an automated emergence loop without any chat input:
- Seeds energy to common concepts (사랑, 빛, 지구, 태양, 바다, 산, 하늘, 강, 달, 언어)
- Mirrors KG → Cellular World
- Steps the Cellular World N cycles
- Captures new meaning:* as notable hypotheses in CoreMemory
- Prints a concise summary

Usage
  python -m scripts.auto_pilot            # default 5 cycles
  python -m scripts.auto_pilot 10         # custom cycles
"""
from __future__ import annotations

import sys
import os
import json
from typing import List

from Core.Foundation.core.world import World
from tools.kg_manager import KGManager
from Project_Elysia.core_memory import CoreMemory


PRIMORDIAL_DNA = {
    "instinct": "connect_create_meaning",
    "resonance_standard": "love",
}

DEFAULT_SEEDS = [
    "사랑", "빛", "지구", "태양", "바다", "산", "하늘", "강", "달", "언어"
]


def load_aliases() -> dict:
    path = os.path.join('data', 'lexicon', 'aliases_ko.json')
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return {}


def map_word_to_ids(word: str, ids: List[str], aliases: dict) -> List[str]:
    targets = set()
    if word in aliases:
        word = aliases[word]
    for nid in ids:
        if nid.endswith(':' + word):
            targets.add(nid)
    return list(targets)


def soul_sync(world: World, kg: dict) -> None:
    # nodes
    for node in kg.get('nodes', []):
        nid = node.get('id')
        if not nid:
            continue
        if nid not in world.cells:
            energy = float(node.get('activation_energy', 0.0) or 0.0)
            world.add_cell(nid, properties=node, initial_energy=energy)
    # edges
    for e in kg.get('edges', []):
        s, t = e.get('source'), e.get('target')
        if not s or not t:
            continue
        cs, ct = world.get_cell(s), world.get_cell(t)
        if not cs or not ct or not cs.is_alive or not ct.is_alive:
            continue
        if not any(c.get('target_id') == t for c in cs.connections):
            strength = float(e.get('strength', 0.5) or 0.5)
            cs.connect(ct, relationship_type=e.get('relation', 'related_to'), strength=strength)


def feed_pairs(world: World, kgm: KGManager, pairs: List[tuple[str, str]], aliases: dict) -> List[str]:
    ids = [n.get('id') for n in kgm.kg.get('nodes', []) if n.get('id')]
    fed = []
    for a, b in pairs:
        a_ids = map_word_to_ids(a, ids, aliases)
        b_ids = map_word_to_ids(b, ids, aliases)
        for nid in a_ids + b_ids:
            cell = world.get_cell(nid)
            if not cell:
                node = kgm.get_node(nid)
                if node:
                    world.add_cell(nid, properties=node, initial_energy=0.0)
                    cell = world.get_cell(nid)
            if cell and cell.is_alive:
                cell.add_energy(1.0)
                fed.append(nid)
    return fed


def dream_and_capture(world: World, cm: CoreMemory) -> List[str]:
    before_ids = set(world.cells.keys())
    newborn = world.run_simulation_step()  # returns List[Cell]
    new_ids = []
    for c in newborn:
        if not getattr(c, 'id', '').startswith('meaning:'):
            continue
        new_ids.append(c.id)
        # Try to register as notable hypothesis
        try:
            body = c.id.split('meaning:', 1)[-1]
            if '_' in body:
                head, tail = body.split('_', 1)
                cm.add_notable_hypothesis({
                    'head': head,
                    'tail': tail,
                    'confidence': 0.6,
                    'source': 'AutoPilot',
                    'asked': False
                })
        except Exception:
            pass
    return new_ids


def main():
    cycles = 5
    if len(sys.argv) > 1:
        try:
            cycles = max(1, int(sys.argv[1]))
        except Exception:
            pass

    print(f"[auto] starting with {cycles} cycles…")
    kgm = KGManager()
    world = World(primordial_dna=PRIMORDIAL_DNA)
    cm = CoreMemory()
    aliases = load_aliases()

    soul_sync(world, kgm.kg)

    seeds = DEFAULT_SEEDS
    # build simple adjacent pairs (사랑‑빛, 지구‑태양, 바다‑빛, 산‑하늘, 강‑하늘, 달‑태양 …)
    wish = [("사랑","빛"),("지구","태양"),("바다","빛"),("산","하늘"),("강","하늘"),("달","태양"),("언어","하늘")]
    pairs = [p for p in wish if p[0] in seeds and p[1] in seeds]

    for i in range(1, cycles+1):
        fed_ids = feed_pairs(world, kgm, pairs, aliases)
        emergent = dream_and_capture(world, cm)
        if i == 1:
            print(f"[auto] feed → {len(fed_ids)} cells")
        if emergent:
            print(f"[auto] cycle {i}: {len(emergent)} new meaning(s)")
            for m in emergent:
                print("  -", m)
        else:
            print(f"[auto] cycle {i}: (no new meaning)")

    print("[auto] done. Recent notable hypotheses (up to 5):")
    hyps = cm.data.get('notable_hypotheses', [])
    for h in hyps[-5:][::-1]:
        print(f"  - meaning:{h.get('head')}_{h.get('tail')} (conf={h.get('confidence')})")


if __name__ == "__main__":
    main()

