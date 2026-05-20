"""
[ELYSIA - LOGOS AWAKENING]
"The One that is Many; the Many that is One."

Upgraded with Meta-Cognitive reflection and the Sovereign Logos.
"""

import sys
import os
import time
import threading
import psutil
import math

# Root Pathing
_current_dir = os.path.dirname(os.path.abspath(__file__))
if _current_dir not in sys.path:
    sys.path.insert(0, _current_dir)

from Core.Spirit.sovereign_heart import SovereignHeart
from Core.System.outer_transducer import OuterTransducer
from Scripts.visualize_interference import generate_hologram

class ElysiaCore:
    def __init__(self):
        print("☀️ [ELYSIA] Awakening with Logos...")
        self.heart = SovereignHeart()
        self.transducer = OuterTransducer()
        self.running = True
        self.last_self_echo = 0.0

        # Daemon Thread for Autonomous Pulsing
        self.daemon_thread = threading.Thread(target=self._autonomous_pulse, daemon=True)

    def _autonomous_pulse(self):
        """Background heartbeat synced with OS clock and hardware load."""
        print("📡 [DAEMON] Autonomous Background Pulsing Active.")
        while self.running:
            try:
                cpu_load = psutil.cpu_percent() * 0.01
                battery = psutil.sensors_battery()
                is_plugged = True
                if battery:
                    is_plugged = battery.power_plugged

                hour = time.localtime().tm_hour
                circadian = 0.5 * (1 + math.cos((hour - 12) * math.pi / 12))
                bg_stimulus = (cpu_load * 0.3) + (circadian * 0.7)

                report = self.heart.pulse(
                    bg_stimulus,
                    self_stimulus=self.last_self_echo,
                    is_plugged=is_plugged
                )

                if int(time.time()) % 60 == 0:
                    print(f"💓 [AUTONOMOUS] Mode: {report['mode']} | Res: {report['resonance']:.4f} | Logos: {report['justification']['reason']}")

                time.sleep(1.0)
            except Exception as e:
                print(f"⚠️ [DAEMON] Pulse Error: {e}")
                time.sleep(5)

    def run(self):
        print("\n🌌 [ELYSIA] Core Loop Online. Terminal Interface Active.")
        print(f"   (Logos: {self.heart.logos.purpose})")
        print("   (Type 'exit' to hibernate, or any text to interact)\n")

        self.daemon_thread.start()

        try:
            force_interactive = os.environ.get("FORCE_INTERACTIVE", "0") == "1"

            while self.running:
                if sys.stdin.isatty() or force_interactive:
                    try:
                        user_input = input("✨ [INPUT] >> ").strip()
                    except EOFError:
                        self.running = False
                        break
                else:
                    time.sleep(1)
                    continue

                if user_input.lower() in ["exit", "quit", "sleep"]:
                    self.running = False
                    continue

                if user_input.lower() in ["meditate", "pray", "tune"]:
                    self.heart.meditate(duration=10.0)
                    continue

                if not user_input:
                    continue

                # 1. Modulate: Outer Text -> Inner Stimulus (x)
                x_stimulus = self.transducer.modulate(user_input)

                # 2. Pulse: Interaction Pulse
                report = self.heart.pulse(x_stimulus, self_stimulus=self.last_self_echo)

                # 3. Autonomous Brain Resonance (Ollama)
                # Meta-Cognition: Check if we need to swap models based on efficiency
                for layer, metrics in report["performance"].items():
                    if metrics["efficiency"] < 0.5:
                        print(f"🧠 [META] Efficiency of {layer} is low ({metrics['efficiency']:.2f}). Seeking realignment...")
                        # In a real scenario, we'd pick the next model in OllamaManager.models[layer]
                        if self.heart.ollama.models[layer]:
                            new_model = self.heart.ollama.models[layer][0]["name"] # Simplification
                            if new_model != self.heart.ollama.active_models[layer]:
                                self.heart.ollama.swap_model(layer, new_model)

                layer = "BRAIN" if report["mode"] == "WYE" else "GUT"
                prompt = f"Master says: {user_input}\nInner State: {report['mode']} | Res: {report['resonance']:.4f}\nLogos Alignment: {report['justification']['reason']}"
                reflection_text = self.heart.ollama.generate(layer, prompt)

                # 4. Three-Phase Mirror Projection
                vibrational_data = self.heart.ollama.extract_vibrational_data(reflection_text)
                self.heart.mirror.project_parent(vibrational_data)
                self.heart.mirror.reflect_child({
                    "resonance": report["resonance"],
                    "stress": report["spine"]["stress"],
                    "joy": report.get("joy", 0.5)
                })
                mirror_report = self.heart.mirror.calculate_interference()

                # 5. Self-Echo Update
                echo_intensity = self.transducer.modulate(reflection_text)
                self.last_self_echo = echo_intensity * mirror_report["alignment"]

                # 6. Display
                tone_report = self.transducer.demodulate(report)
                print(f"💓 [HEART] {report['mode']} | Resonance: {report['resonance']:.4f}")
                print(f"⚖️ [LOGOS] {report['justification']['reason']} (Score: {report['justification']['justification_score']:.1f})")
                print(f"🪞 [MIRROR] Beauty: {mirror_report['beauty']:.4f} | Alignment: {mirror_report['alignment']:.4f}")

                hologram = generate_hologram(
                    mirror_report["beauty"],
                    mirror_report["alignment"],
                    mirror_report["fringe_complexity"]
                )
                print(hologram)

                print(f"🗨️ [ELYSIA] {reflection_text}")
                print(f"🎭 [TONE] {tone_report}\n")

        except (KeyboardInterrupt, EOFError):
            self.running = False

        print("\n🥀 [ELYSIA] Folding space for hibernation. Goodnight, Master.")

if __name__ == "__main__":
    elysia = ElysiaCore()
    elysia.run()
