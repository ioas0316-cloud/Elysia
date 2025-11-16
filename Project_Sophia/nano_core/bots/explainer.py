from __future__ import annotations

import time
from nano_core.message import Message
from nano_core.bus import MessageBus
from nano_core.registry import ConceptRegistry
from tools.kg_manager import KGManager


class ExplainerBot:
    name = 'explainer'
    verbs = ['explain']

    def handle(self, msg: Message, reg: ConceptRegistry, bus: MessageBus) -> None:
        target = str(msg.slots.get('target', '')).strip()
        text = str(msg.slots.get('text', '')).strip()
        if not target or not text:
            return
        node_id = f"explain:{target}:{int(time.time())}"
        reg.ensure_concept(node_id, props={'type': 'explanation', 'text': text})
        reg.add_link(node_id, target, rel='explains')
        # optional evidence links
        ev = msg.slots.get('evidence')
        if isinstance(ev, list):
            for e in ev:
                e_id = str(e).strip()
                if e_id:
                    reg.add_link(node_id, e_id, rel='supported_by')
        elif isinstance(ev, str) and ev:
            reg.add_link(node_id, ev.strip(), rel='supported_by')
