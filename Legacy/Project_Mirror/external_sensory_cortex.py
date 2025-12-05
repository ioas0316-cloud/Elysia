import logging
import platform
import os
try:
    import psutil
except ImportError:
    psutil = None

from typing import Any, Dict, Optional
from Core.Foundation.core.external_horizons import ExternalHorizon
from Core.Foundation.web_search_cortex import WebSearchCortex

logger = logging.getLogger(__name__)

class ExternalSensoryCortex:
    """
    The 'Periscope' for the External World (Y-Axis).
    Switches focus between the 7 Horizons based on the intensity of the External Focus.
    """

    def __init__(self, web_search_cortex: Optional[WebSearchCortex] = None):
        self.web_search_cortex = web_search_cortex

    def sense(self, horizon: ExternalHorizon, intensity: float) -> Dict[str, Any]:
        """
        Performs a sensory scan of the specified horizon.
        """
        logger.info(f"EXTERNAL SENSE: Focusing on Horizon {horizon.name} (Level {horizon.value})")

        if horizon == ExternalHorizon.MACHINE:
            return self._sense_machine()
        elif horizon == ExternalHorizon.SHELL:
            return self._sense_shell()
        elif horizon == ExternalHorizon.INTERFACE:
            return self._sense_interface()
        elif horizon == ExternalHorizon.NETWORK:
            return self._sense_network()
        elif horizon == ExternalHorizon.WEB:
            return self._sense_web(intensity)
        elif horizon == ExternalHorizon.ZEITGEIST:
            return self._sense_zeitgeist()
        elif horizon == ExternalHorizon.REALITY:
            return self._sense_reality()

        return {"error": "Unknown Horizon"}

    def _sense_machine(self) -> Dict[str, Any]:
        """Horizon 1: The Machine (Body Sensation)."""
        data = {
            "system": platform.system(),
            "processor": platform.processor(),
        }
        if psutil:
            data.update({
                "cpu_percent": psutil.cpu_percent(interval=0.1),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_percent": psutil.disk_usage('/').percent
            })
        return {"type": "sensation_machine", "data": data}

    def _sense_shell(self) -> Dict[str, Any]:
        """Horizon 2: The Shell (Environment)."""
        return {
            "type": "sensation_shell",
            "data": {
                "cwd": os.getcwd(),
                "pid": os.getpid(),
                "files_in_root": len(os.listdir('.'))
            }
        }

    def _sense_interface(self) -> Dict[str, Any]:
        """Horizon 3: The Interface."""
        # Placeholder for checking chat logs or screen buffers
        return {"type": "sensation_interface", "status": "active_connection_inferred"}

    def _sense_network(self) -> Dict[str, Any]:
        """Horizon 4: The Network."""
        # Placeholder
        return {"type": "sensation_network", "status": "connected"}

    def _sense_web(self, intensity: float) -> Dict[str, Any]:
        """Horizon 5: The Web."""
        if not self.web_search_cortex:
            return {"type": "sensation_web", "status": "cortex_missing"}

        # If intensity is high, actually perform a search (Curiosity)
        # For passive sensing, we might just check connectivity or headlines.
        return {"type": "sensation_web", "status": "ready_to_search"}

    def _sense_zeitgeist(self) -> Dict[str, Any]:
        """Horizon 6: The Zeitgeist."""
        # Placeholder for "Trending Topics"
        return {"type": "sensation_zeitgeist", "status": "unknown"}

    def _sense_reality(self) -> Dict[str, Any]:
        """Horizon 7: The User's Reality."""
        # This is the deepest level of empathy.
        # It attempts to read the 'Context' of the user (e.g., "Kimjang", "Countryside").
        return {
            "type": "sensation_reality",
            "focus": "user_context",
            "note": "Connecting to User's physical and emotional location."
        }
