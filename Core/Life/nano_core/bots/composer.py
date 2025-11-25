from __future__ import annotations

import re
from nano_core.message import Message
from nano_core.bus import MessageBus
from nano_core.registry import ConceptRegistry
from tools.kg_manager import KGManager


def _sanitize(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9:_|+-]", "_", s)


class ComposerBot:
    name = 'composer'
    verbs = ['compose']

    def handle(self, msg: Message, reg: ConceptRegistry, bus: MessageBus) -> None:
        a = str(msg.slots.get('a', '')).strip()
        b = str(msg.slots.get('b', '')).strip()
        if not a or not b:
            return
        combo_id = f"concept:combo:{_sanitize(a)}|{_sanitize(b)}"
        reg.ensure_concept(combo_id, props={'type': 'concept_combo'})
        # prevent duplicate edges
        try:
            kg: KGManager = reg.kg
            edges = {(e.get('source'), e.get('target'), e.get('relation')) for e in kg.kg.get('edges', [])}
        except Exception:
            edges = set()
        if (combo_id, a, 'composed_of') not in edges:
            reg.add_link(combo_id, a, rel='composed_of')
        if (combo_id, b, 'composed_of') not in edges:
            reg.add_link(combo_id, b, rel='composed_of')
        # Optional bridging relation
        rel2 = str(msg.slots.get('rel2', '')).strip()
        if rel2:
            if (a, b, rel2) not in edges:
                reg.add_link(a, b, rel=rel2)
