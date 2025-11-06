from __future__ import annotations

from nano_core.message import Message
from nano_core.bus import MessageBus
from nano_core.registry import ConceptRegistry
from nano_core.telemetry import write_event


class SummarizerBot:
    name = 'summarizer'
    verbs = ['summarize']

    def handle(self, msg: Message, reg: ConceptRegistry, bus: MessageBus) -> None:
        target = str(msg.slots.get('target', '')).strip()
        if not target:
            return
        kg = reg.kg.kg
        # Collect neighbors (simple 1-hop)
        neigh = []
        for e in kg.get('edges', []):
            if e.get('source') == target:
                neigh.append((e.get('relation', 'related_to'), e.get('target')))
            elif e.get('target') == target:
                neigh.append((e.get('relation', 'related_to'), e.get('source')))
        write_event('concept.summary', {'concept': target, 'neighbors': neigh[:10]})

