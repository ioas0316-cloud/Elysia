from __future__ import annotations

from nano_core.message import Message
from nano_core.bus import MessageBus
from nano_core.registry import ConceptRegistry
from tools.kg_manager import KGManager


class ValidatorBot:
    name = 'validator'
    verbs = ['verify']

    def handle(self, msg: Message, reg: ConceptRegistry, bus: MessageBus) -> None:
        subj = str(msg.slots.get('subject', '')).strip()
        obj = str(msg.slots.get('object', '')).strip()
        rel = str(msg.slots.get('rel', 'related_to'))
        ok = False
        try:
            kg: KGManager = reg.kg
            for e in kg.kg.get('edges', []):
                if e.get('source') == subj and e.get('target') == obj and e.get('relation') == rel:
                    ok = True
                    break
        except Exception:
            pass
        # If missing, schedule a link with lower strength
        if not ok and subj and obj:
            bus.post(Message(verb='link', slots={'subject': subj, 'object': obj, 'rel': rel}, strength=max(0.5, msg.strength * 0.8), ttl=max(1, msg.ttl - 1)))

