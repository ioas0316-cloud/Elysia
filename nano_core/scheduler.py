from __future__ import annotations

from typing import Dict, List, Protocol
from nano_core.message import Message
from nano_core.bus import MessageBus
from nano_core.registry import ConceptRegistry
from nano_core.telemetry import write_event


class Bot(Protocol):
    name: str
    verbs: List[str]
    def handle(self, msg: Message, reg: ConceptRegistry, bus: MessageBus) -> None: ...


class Scheduler:
    def __init__(self, bus: MessageBus, registry: ConceptRegistry, bots: List[Bot]) -> None:
        self.bus = bus
        self.registry = registry
        self._router: Dict[str, Bot] = {}
        for b in bots:
            for v in b.verbs:
                self._router[v] = b

    def step(self, max_steps: int = 50) -> int:
        processed = 0
        while processed < max_steps and not self.bus.empty():
            msg = self.bus.get_next()
            if not msg:
                break
            if msg.ttl <= 0:
                continue
            bot = self._router.get(msg.verb)
            write_event('bot.run', {'bot': getattr(bot, 'name', 'unknown'), 'verb': msg.verb, 'id': msg.id})
            try:
                if bot:
                    bot.handle(msg, self.registry, self.bus)
                msg.decay()
                processed += 1
            except Exception as e:
                write_event('bot.run', {'bot': getattr(bot, 'name', 'unknown'), 'verb': msg.verb, 'id': msg.id, 'error': str(e)})
        # Persist KG after a burst
        try:
            self.registry.save()
        except Exception:
            pass
        return processed

