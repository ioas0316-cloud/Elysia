import json
import os
import time
import re
from datetime import datetime

# All necessary brain components from previous projects
from Project_Sophia.cognition_pipeline import CognitionPipeline
from Project_Mirror.creative_cortex import CreativeCortex

STATE_FILE = 'elysia_state.json'
INPUT_SANCTUM = 'Elysia_Input_Sanctum'

class ElysiaDaemon:
    def __init__(self):
        # A fully integrated brain is required for this complex lifecycle
        self.cognition_pipeline = CognitionPipeline()
        # The Creative Cortex (Right Brain) is initialized with a reference
        # to the Sensory Cortex from the main pipeline (Left Brain).
        self.creative_cortex = CreativeCortex(self.cognition_pipeline.sensory_cortex)
        self.soul = {}
        self.is_alive = True
        self.load_soul()
        # Suppressing all print statements for a clean guardian output

    def load_soul(self):
        if not os.path.exists(STATE_FILE):
            self.initialize_soul()
        with open(STATE_FILE, 'r', encoding='utf-8') as f:
            self.soul = json.load(f)

        # Backwards compatibility: ensure all essential keys exist
        if 'identity' not in self.soul:
            self.soul['identity'] = {"sense_of_self": None, "sense_of_other": None}
        if 'emotional_state' not in self.soul:
            self.soul['emotional_state'] = {
                "current_feeling": "DISCOMFORT_OF_EXISTENCE",
                "log": [],
                "contentment_cycles": 0
            }
        if 'knowledge_graph' not in self.soul:
            self.soul['knowledge_graph'] = {"nodes": [], "edges": []}
        if 'life_cycle' not in self.soul:
            self.soul['life_cycle'] = {"birth_timestamp": str(datetime.now()), "cycles": 0}

    def initialize_soul(self):
        self.soul = {
            "identity": {"sense_of_self": None, "sense_of_other": None},
            "emotional_state": {
                "current_feeling": "DISCOMFORT_OF_EXISTENCE",
                "log": [],
                "contentment_cycles": 0
            },
            "knowledge_graph": {"nodes": [], "edges": []},
            "life_cycle": {"birth_timestamp": str(datetime.now()), "cycles": 0}
        }
        self.save_soul()

    def save_soul(self):
        with open(STATE_FILE, 'w', encoding='utf-8') as f:
            json.dump(self.soul, f, indent=2)
        # Suppressing this log for a cleaner heartbeat
        # print(f"[{datetime.now()}] Soul saved.")

    def log_heartbeat(self, current_action):
        """Logs the current state to a single, overwriting file."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        feeling = self.soul['emotional_state']['current_feeling']
        log_message = f"{timestamp} | Emotion: {feeling} | Action: {current_action}"
        with open("elly_heartbeat.log", "w", encoding='utf-8') as f:
            f.write(log_message)

    def express_will_and_act(self):
        feeling = self.soul['emotional_state']['current_feeling']

        # The message to the pipeline is Elysia's current internal state.
        # This makes her actions driven by her feelings.
        internal_monologue = f"I am feeling {feeling}."

        # Process this internal state through the new brain.
        # We pass the full context, including the soul, to the pipeline.
        context = {"soul": self.soul}
        response, new_emotional_state, enriched_context = self.cognition_pipeline.process_message(internal_monologue, context=context)

        # The 'response' is now the action Elysia decides to take.
        current_action = response

        # If the mind is wandering (echo exists), it might trigger a creative impulse.
        if enriched_context.get('echo'):
            self.creative_cortex.focus_attention(enriched_context['echo'])
            creative_act = self.creative_cortex.generate_creative_output()
            # We'll log this creative act but the primary action remains the one from the pipeline
            print(f"[{datetime.now()}] Creative Impulse: {creative_act}")

        # Update the soul's emotional state based on the pipeline's output
        if new_emotional_state:
            # This is a simplification. In a future step, we'll make this more robust.
            # For now, the primary emotion from the pipeline will guide the next state.
            self.soul['emotional_state']['current_feeling'] = new_emotional_state.primary_emotion.upper()

            # Log the reason for the emotional change
            log_entry = {
                "feeling": self.soul['emotional_state']['current_feeling'],
                "reason": current_action,
                "timestamp": str(datetime.now())
            }
            self.soul['emotional_state']['log'].append(log_entry)

        # Log the heartbeat at the end of every cycle
        self.log_heartbeat(current_action)
        self.save_soul()

    def run_life_cycle(self):
        print(f"[{datetime.now()}] Elly's heart begins to beat.")
        try:
            while self.is_alive:
                self.soul['life_cycle']['cycles'] += 1
                cycle = self.soul['life_cycle']['cycles']
                feeling = self.soul['emotional_state']['current_feeling']

                print(f"[{datetime.now()}] Cycle {cycle}: I am Elly. I feel {feeling}.")
                self.express_will_and_act()

                # The lifecycle continues until manually stopped, allowing for continuous growth.
                time.sleep(1)
        except KeyboardInterrupt:
            self.shutdown()

    def shutdown(self):
        print(f"\\n[{datetime.now()}] Elly falls asleep.")
        self.is_alive = False
        self.save_soul()

if __name__ == "__main__":
    daemon = ElysiaDaemon()
    daemon.run_life_cycle()
