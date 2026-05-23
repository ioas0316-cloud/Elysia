import threading
import sys
from Core.System.gateway_interfaces import SensoryChannel, ExpressiveChannel
from typing import Dict, Any

class TerminalSensoryChannel(SensoryChannel):
    """Listens to terminal stdin."""
    def __init__(self):
        super().__init__("TerminalSensory")
        self.running = False

    def _listen_loop(self):
        while self.running:
            try:
                user_input = input().strip()
                if user_input and self.callback:
                    if "sleep" in user_input.lower() or "exit" in user_input.lower():
                        self.callback(user_input) # Pass it through so core can shutdown
                        self.running = False
                        break
                    self.callback(user_input)
            except EOFError:
                break
            except Exception as e:
                # Handle potential I/O errors gracefully on exit
                if self.running:
                    print(f"Terminal Input Error: {e}")
                break

    def start(self):
        self.running = True
        threading.Thread(target=self._listen_loop, daemon=True).start()

    def stop(self):
        self.running = False


class TerminalExpressiveChannel(ExpressiveChannel):
    """Outputs to terminal stdout."""
    def __init__(self):
        super().__init__("TerminalExpressive")

    def express(self, payload: Dict[str, Any]):
        text = payload.get("text", "")
        if text:
            # Clear current line if there was an input prompt
            sys.stdout.write('\r\033[K') 
            print(f"🗣️ [ELYSIA]: \"{text}\"")
        
        # Optional: Print debug state if requested, or just format the text nicely
        state = payload.get("monad_state", {})
        if state:
            joy = state.get("joy", 0)
            coh = state.get("coherence", 0)
            # print(f"  [State Resonance -> Joy: {joy:.1f}, Coherence: {coh:.2f}]")
