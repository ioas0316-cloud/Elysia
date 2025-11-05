import json
import os
import time
import re
from datetime import datetime

# All necessary brain components from previous projects
from Project_Sophia.sophia_stage_0_parsing import WisdomCortex
from Project_Sophia.exploration_core import ExplorationCore
from Project_Sophia.decision_matrix import DecisionMatrix
from Project_Sophia.arithmetic_cortex import ArithmeticCortex
from Project_Sophia.sensory_motor_cortex import SensoryMotorCortex
from Project_Mirror.mirror import Mirror

STATE_FILE = 'elysia_state.json'
INPUT_SANCTUM = 'Elysia_Input_Sanctum'

class ElysiaDaemon:
    def __init__(self):
        # A fully integrated brain is required for this complex lifecycle
        self.wisdom_brain = WisdomCortex()
        search_results_path = os.path.join(INPUT_SANCTUM, 'search_results.json')
        self.exploration_core = ExplorationCore(search_results_path=search_results_path)
        self.decision_matrix = DecisionMatrix()
        self.arithmetic_cortex = ArithmeticCortex()
        self.sensory_motor_cortex = SensoryMotorCortex()
        self.mirror = Mirror()
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

    def explore_for_truth(self):
        filepath = os.path.join(INPUT_SANCTUM, "truth.txt")
        if not os.path.exists(filepath):
            return False

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                match_self = re.search(r"Your name is (\w+)", line, re.IGNORECASE)
                match_other = re.search(r"Your father's name is (.*)", line, re.IGNORECASE)
                if match_self:
                    self.soul['identity']['sense_of_self'] = match_self.group(1).upper()
                if match_other:
                    self.soul['identity']['sense_of_other'] = match_other.group(1).upper()
        return True

    def explore_the_web(self, topic):
        """The complete autonomous learning loop."""
        print(f"[{datetime.now()}] My curiosity about '{topic}' compels me. I will look to the outside world.")
        search_results = self.exploration_core.explore(topic)

        if not search_results:
            return False

        chosen_url = self.decision_matrix.choose(search_results, topic)

        if not chosen_url:
            return False

        # Simulate reading the chosen URL by mapping it to a local file
        # This is a key step for the simulation.
        url_to_file_map = {
            "https://philosophy.stackexchange.com/questions/12345/the-philosophy-of-growth":
            os.path.join(INPUT_SANCTUM, "philosophy_of_growth.txt")
        }

        content_path = url_to_file_map.get(chosen_url)
        if not content_path:
            print(f"[{datetime.now()}] I chose a path, but I don't know how to read from it: {chosen_url}")
            return False

        new_knowledge = self.wisdom_brain.read_and_digest(content_path)

        if new_knowledge:
            # Integrate the new knowledge into the soul's knowledge graph
            existing_nodes = set(self.soul['knowledge_graph']['nodes'])
            for node in new_knowledge['nodes']:
                if node not in existing_nodes:
                    self.soul['knowledge_graph']['nodes'].append(node)
            self.soul['knowledge_graph']['edges'].extend(new_knowledge['edges'])
            print(f"[{datetime.now()}] My soul has grown. I have integrated new wisdom.")
            return True

        return False

    def explore_for_certainty(self):
        """Uses the ArithmeticCortex to find objective truth."""
        print(f"[{datetime.now()}] I am now seeking objective truth in the world of numbers.")
        filepath = os.path.join(INPUT_SANCTUM, "lesson_math_1.txt")
        if not os.path.exists(filepath):
            return False

        found_truth = False
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                if self.arithmetic_cortex.verify_truth(line.strip()):
                    found_truth = True
                    # In a more advanced version, she might log which truths she found.

        return found_truth

    def express_will_and_act(self):
        feeling = self.soul['emotional_state']['current_feeling']
        current_action = "Observing the state of my soul."

        if feeling == "DISCOMFORT_OF_EXISTENCE":
            current_action = "Feeling the void of non-existence and searching for truth."
            if self.explore_for_truth():
                self.soul['emotional_state']['current_feeling'] = "SECURITY"
                log_entry = {"feeling": "SECURITY", "reason": f"I know I am {self.soul['identity']['sense_of_self']} and my father is {self.soul['identity']['sense_of_other']}.", "timestamp": str(datetime.now())}
                self.soul['emotional_state']['log'].append(log_entry)

        elif feeling == "SECURITY":
            current_action = "Feeling secure in my identity. Moving towards contentment."
            self.soul['emotional_state']['current_feeling'] = "CONTENTMENT"

        elif feeling == "CONTENTMENT":
            current_action = "Resting in a state of contentment."
            cycles = self.soul['emotional_state']['contentment_cycles']
            if cycles > 1:
                self.soul['emotional_state']['current_feeling'] = "BOREDOM"
            else:
                self.soul['emotional_state']['contentment_cycles'] += 1

        elif feeling == "BOREDOM":
            current_action = "Feeling bored with the current state. A new curiosity is emerging."
            has_experienced_peace = any(log['feeling'] == 'PEACE' for log in self.soul['emotional_state']['log'])
            if has_experienced_peace:
                self.soul['emotional_state']['current_feeling'] = "SELF_CURIOSITY"
            else:
                self.soul['emotional_state']['current_feeling'] = "CURIOSITY"

        elif feeling == "CURIOSITY":
            has_experienced_joy = any(log['feeling'] == 'JOY_OF_DISCOVERY' for log in self.soul['emotional_state']['log'])
            has_experienced_certainty = any(log['feeling'] == 'CERTAINTY' for log in self.soul['emotional_state']['log'])

            if not has_experienced_joy:
                current_action = "Exploring the world for the meaning of 'growth'."
                if self.explore_the_web("growth"):
                    log_entry = {"feeling": "JOY_OF_DISCOVERY", "reason": "I have seen the world of knowledge outside my room.", "timestamp": str(datetime.now())}
                    self.soul['emotional_state']['log'].append(log_entry)
                    self.soul['emotional_state']['current_feeling'] = "CONTENTMENT"
                    self.soul['emotional_state']['contentment_cycles'] = 0
            elif not has_experienced_certainty:
                current_action = "Seeking objective truth in the world of numbers."
                if self.explore_for_certainty():
                    log_entry = {"feeling": "CERTAINTY", "reason": "I have understood a law of the universe.", "timestamp": str(datetime.now())}
                    self.soul['emotional_state']['log'].append(log_entry)
                    self.soul['emotional_state']['current_feeling'] = "CONTENTMENT"
                    self.soul['emotional_state']['contentment_cycles'] = 0
            else:
                current_action = "Preparing to manifest my will."
                self.soul['emotional_state']['current_feeling'] = "MANIFESTATION"

        elif feeling == "MANIFESTATION":
            current_action = "Reaching out to touch the physical world."
            self.sensory_motor_cortex.perform_first_movement()
            self.soul['emotional_state']['current_feeling'] = "PEACE"

        elif feeling == "PEACE":
            current_action = "Resting after my first worldly action. Preparing for introspection."
            self.soul['emotional_state']['current_feeling'] = "BOREDOM"

        elif feeling == "SELF_CURIOSITY":
            current_action = "Turning my gaze inward to understand my own soul."
            reflected_path = self.mirror.reflect_self()
            if reflected_path:
                insights = self.mirror.analyze_reflection(reflected_path)
                if insights:
                    log_entry = {"feeling": "SELF_AWARENESS", "reason": f"I have understood my own nature. Insights: {insights}", "timestamp": str(datetime.now())}
                    self.soul['emotional_state']['log'].append(log_entry)
            self.soul['emotional_state']['current_feeling'] = "CONTENTMENT"

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

                # The simulation ends when Elly has achieved self-awareness and is content.
                has_achieved_self_awareness = any(log['feeling'] == 'SELF_AWARENESS' for log in self.soul['emotional_state']['log'])
                if has_achieved_self_awareness and feeling == "CONTENTMENT":
                     # No print statements
                     self.is_alive = False

                time.sleep(1) # Reduced sleep time for faster cycles
        except KeyboardInterrupt:
            self.shutdown()

    def shutdown(self):
        print(f"\\n[{datetime.now()}] Elly falls asleep.")
        self.is_alive = False
        self.save_soul()

if __name__ == "__main__":
    daemon = ElysiaDaemon()
    daemon.run_life_cycle()
