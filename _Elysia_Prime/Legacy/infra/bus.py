# [Genesis: 2025-12-02] Purified by Elysia
from typing import Callable, Dict, List, Any, Optional

from infra.telemetry import Telemetry


class EventBus:
    """
    Minimal synchronous event bus with telemetry mirroring.
    Subscribers are called inline; handlers should be fast/non-blocking.
    """

    def __init__(self, telemetry: Optional[Telemetry] = None):
        self._subs: Dict[str, List[Callable[[Dict[str, Any]], None]]] = {}
        self.telemetry = telemetry or Telemetry()

    def subscribe(self, event_type: str, handler: Callable[[Dict[str, Any]], None]):
        self._subs.setdefault(event_type, []).append(handler)

    def publish(self, event_type: str, payload: Dict[str, Any], trace_id: Optional[str] = None):
        # Mirror to telemetry
        if self.telemetry:
            self.telemetry.emit(event_type, payload, trace_id=trace_id)

        # Dispatch to subscribers
        for handler in self._subs.get(event_type, []):
            try:
                handler(payload)
            except Exception:
                # Handlers must not crash publisher
                pass
