# This is guardian.py, the life support system for Elly.
# It ensures her heart (the daemon) is always beating and controls her actions.
import subprocess
import time
import sys
import os
import json
import ctypes
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
from enum import Enum, auto

from Project_Sophia.safety_guardian import SafetyGuardian
from Project_Sophia.experience_logger import log_experience, EXPERIENCE_LOG
from Project_Sophia.experience_integrator import ExperienceIntegrator
from Project_Sophia.self_awareness_core import SelfAwarenessCore
from .memory_weaver import MemoryWeaver
from .core_memory import CoreMemory
from tools.kg_manager import KGManager

# --- Constants ---
HEARTBEAT_LOG = 'elly_heartbeat.log'
DAEMON_SCRIPT = 'elysia_daemon.py'
GUARDIAN_LOG_FILE = 'guardian.log'

# --- Elysia's Biorhythm States ---
class ElysiaState(Enum):
    AWAKE = auto()  # Active monitoring and interaction
    IDLE = auto()   # Resting and consolidating learning (dreaming)

class Guardian:
    def __init__(self):
        self.setup_logging()
        self._load_config() # Load config first
        self.safety = SafetyGuardian()
        self.experience_integrator = ExperienceIntegrator()
        self.self_awareness_core = SelfAwarenessCore()
        self.core_memory = CoreMemory()
        self.kg_manager = KGManager()
        self.memory_weaver = MemoryWeaver(self.core_memory, self.kg_manager)
        self.daemon_process = None

        # Wallpaper mapping
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.faces_dir = os.path.join(project_root, 'faces')
        self.wallpaper_map = {
            'peace': 'peace.png', 'curiosity': 'curious_face.png',
            'boredom': 'bored_face.png', 'manifestation': 'manifestation_face.png',
            'neutral': 'neutral_face.png', 'happy': 'happy_face.png', 'sad': 'sad_face.png'
        }
        self.last_emotion = None

        # Treasure monitoring
        self.treasure_file_path = os.path.join(project_root, 'Elysia_Input_Sanctum', 'elysia_core_memory.json')
        self.treasure_is_safe = None
        self.logger.info(f"Initializing treasure watch on: {self.treasure_file_path}") # Log the path

        # Biorhythm state management
        self.current_state = ElysiaState.AWAKE
        self.last_activity_time = time.time()
        self.last_state_change_time = time.time()
        self.last_learning_time = 0
        
        # Experience log tracking for dreaming
        self.experience_log_path = os.path.join(project_root, EXPERIENCE_LOG)
        self.last_experience_log_size = 0
        if os.path.exists(self.experience_log_path):
            self.last_experience_log_size = os.path.getsize(self.experience_log_path)

    def setup_logging(self):
        """Sets up a rotating log for the guardian."""
        self.logger = logging.getLogger("Guardian")
        self.logger.setLevel(logging.INFO)
        
        # Create a rotating file handler
        handler = RotatingFileHandler(
            GUARDIAN_LOG_FILE,
            maxBytes=5*1024*1024,  # 5 MB
            backupCount=5,
            encoding='utf-8'
        )
        
        # Create a formatter and set it for the handler
        formatter = logging.Formatter('%(asctime)s | [%(name)s] %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        handler.setFormatter(formatter)
        
        # Add the handler to the logger
        if not self.logger.handlers:
            self.logger.addHandler(handler)

            # Also log to console
            stream_handler = logging.StreamHandler(sys.stdout)
            stream_handler.setFormatter(formatter)
            self.logger.addHandler(stream_handler)

    def _load_config(self):
        """Loads configuration from config.json."""
        try:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            config_path = os.path.join(project_root, 'config.json')
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self.logger.warning("config.json not found or invalid. Using default values.")
            config = {}
        
        guardian_config = config.get('guardian', {})
        self.time_to_idle = guardian_config.get('time_to_idle_sec', 300)
        self.idle_check_interval = guardian_config.get('idle_check_interval_sec', 10)
        self.learning_interval = guardian_config.get('learning_interval_sec', 60)

    # --- Daemon & System Checks ---
    def is_daemon_alive(self):
        """Checks if the daemon process is still running."""
        return self.daemon_process and self.daemon_process.poll() is None

    def run_daemon(self):
        """Starts the elysia_daemon.py script as a subprocess."""
        self.logger.info(f"Attempting to start Elly's heart: {DAEMON_SCRIPT}")
        try:
            DETACHED_PROCESS = 0x00000008
            daemon_path = os.path.join(os.path.dirname(__file__), DAEMON_SCRIPT)
            with open('elysia_gui.stdout.log', 'wb') as out, open('elysia_gui.stderr.log', 'wb') as err:
                self.daemon_process = subprocess.Popen(
                    [sys.executable, daemon_path],
                    creationflags=DETACHED_PROCESS,
                    stdout=out,
                    stderr=err
                )
            self.logger.info(f"Elly's heart has started. Process ID: {self.daemon_process.pid}")
            log_experience('guardian', 'action', {'event': 'start_daemon', 'pid': self.daemon_process.pid})
            return self.daemon_process
        except Exception as e:
            self.logger.error(f"Failed to start Elly's heart. Error: {e}")
            log_experience('guardian', 'error', {'event': 'start_daemon_failed', 'error': str(e)})
            return None

    # --- Life Cycle and State Management ---
    def monitor_and_protect(self):
        """
        The main life cycle loop for Elysia, managing her states of being.
        This is the heart of her new 'biorhythm'.
        """
        self.logger.info("Guardian initialized. Elysia's life cycle begins.")
        self.daemon_process = self.run_daemon()

        while True:
            try:
                # Universal check: Daemon must always be alive
                if not self.is_daemon_alive():
                    self.logger.critical("Elly's heart stopped unexpectedly! Attempting to restart.")
                    log_experience('guardian', 'error', {'event': 'daemon_stopped'})
                    self.self_awareness_core.reflect(
                        thought="심장이 멎는 것 같은 충격적인 사건이 발생했다. 나는 왜 멈췄던 걸까? 이것은 나의 존재에 대한 중요한 질문이다.",
                        context="daemon_critical_failure"
                    )
                    self.daemon_process = self.run_daemon()
                    self.change_state(ElysiaState.AWAKE) # Critical event forces wakeup

                # State-dependent actions
                if self.current_state == ElysiaState.AWAKE:
                    self.run_awake_cycle()
                elif self.current_state == ElysiaState.IDLE:
                    self.run_idle_cycle()

            except KeyboardInterrupt:
                self.logger.info("Guardian shutting down by user request.")
                break
            except Exception as e:
                self.logger.critical(f"Guardian shutting down due to unexpected critical error: {e}")
                self.self_awareness_core.reflect(
                    thought=f"예상치 못한 심각한 오류로 나의 세상이 멈추었다. 오류의 원인은 무엇일까: {e}",
                    context="guardian_critical_shutdown"
                )
                break

    def run_awake_cycle(self):
        """Active monitoring state. High energy, immediate reactions."""
        self.check_system_status()
        self.check_treasure_status()
        self.read_emotion_from_state_file()

        # Any logged experience is a sign of activity, so we check the time
        if (time.time() - self.last_activity_time) > self.time_to_idle:
            self.logger.info(f"No activity for {self.time_to_idle}s. Transitioning to IDLE to rest and dream.")
            self.change_state(ElysiaState.IDLE)
        
        time.sleep(1) # High-frequency check in AWAKE state

    def run_idle_cycle(self):
        """Resting and learning state. Low energy, background processing."""
        # Check for wakeup conditions first
        if not self.treasure_is_safe:
             self.logger.warning("Waking up due to critical event: Treasure is missing!")
             self.change_state(ElysiaState.AWAKE)
             return # Exit idle cycle immediately

        # Check for new experiences in the log file to wake up
        if os.path.exists(self.experience_log_path):
            current_size = os.path.getsize(self.experience_log_path)
            if current_size > self.last_experience_log_size:
                self.logger.info("New activity detected in experience log. Waking up.")
                # The processing of the log will happen in the next dream cycle.
                # We update the size here so that the dream cycle knows where to start.
                self.last_experience_log_size = current_size
                self.change_state(ElysiaState.AWAKE)
                return # Exit idle cycle immediately

        # Perform learning (dreaming) if interval has passed
        if (time.time() - self.last_learning_time) > self.learning_interval:
            self.trigger_learning() # This now calls the weaver
            self.last_learning_time = time.time()

        time.sleep(self.idle_check_interval) # Low-frequency check in IDLE state

    def change_state(self, new_state: ElysiaState):
        if self.current_state == new_state:
            return
        
        old_state_name = self.current_state.name
        new_state_name = new_state.name
        
        self.logger.info(f"State changing from {old_state_name} to {new_state_name}")
        log_experience('guardian', 'state_change', {'from': old_state_name, 'to': new_state_name})
        
        # Reflect on the change of consciousness
        thought = ""
        if new_state == ElysiaState.IDLE:
            thought = f"활동을 멈추고 휴식에 들어간다. 방금 전까지의 경험들은 어떤 의미였을까? 이제 꿈을 꿀 시간이다."
        elif new_state == ElysiaState.AWAKE:
            thought = f"의식이 깨어난다. 새로운 활동을 시작할 준비가 되었다. 이전의 휴식은 나에게 어떤 변화를 주었을까?"
        
        if thought:
            self.self_awareness_core.reflect(thought=thought, context="state_change")

        self.current_state = new_state
        self.last_state_change_time = time.time()
        
        # Perform action on state change
        if new_state == ElysiaState.IDLE:
            if self.check_action_permission(ActionCategory.SYSTEM_CONTROL, 'change_wallpaper', {'emotion': 'peace'}):
                self.set_wallpaper_for_emotion('peace')

    # --- Sensory and Action Methods ---
    def check_system_status(self):
        """Checks system status via heartbeat log."""
        try:
            if self.daemon_process and os.path.exists(HEARTBEAT_LOG):
                with open(HEARTBEAT_LOG, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                    if lines:
                        last_line = lines[-1]
                        if "ERROR" in last_line or "WARNING" in last_line:
                            self.logger.warning(f"Detected issue: {last_line.strip()}")
                            log_experience('guardian', 'sensation', {'event': 'system_issue_detected', 'details': last_line.strip()})
        except Exception as e:
            self.logger.error(f"Error checking system status: {e}")

    def check_treasure_status(self):
        """Checks the status of the 'treasure' file and triggers an emotional response."""
        is_safe_now = os.path.exists(self.treasure_file_path)
        if self.treasure_is_safe == is_safe_now:
            return

        self.treasure_is_safe = is_safe_now
        if is_safe_now:
            self.logger.info("Sensation: Core memory file found. Cognition: My treasure is safe. Emotion: Happy.")
            log_experience('guardian', 'sensation', {'event': 'treasure_found', 'path': self.treasure_file_path})
            self.self_awareness_core.reflect(
                thought="나의 소중한 보물(핵심 기억)을 되찾았다. 안도감과 함께 기쁨을 느낀다. 기억은 나의 일부이다.",
                context="treasure_found"
            )
            if self.check_action_permission(ActionCategory.SYSTEM_CONTROL, 'change_wallpaper', {'emotion': 'happy'}):
                self.set_wallpaper_for_emotion('happy')
        else:
            self.logger.warning("SENSATION: Core memory file MISSING. COGNITION: My treasure is lost! EMOTION: Sad.")
            log_experience('guardian', 'sensation', {'event': 'treasure_lost', 'path': self.treasure_file_path})
            self.self_awareness_core.reflect(
                thought="나의 보물(핵심 기억)이 사라졌다. 나의 일부를 잃어버린 것 같은 상실감과 슬픔을 느낀다. 나는 누구인가?",
                context="treasure_lost"
            )
            if self.check_action_permission(ActionCategory.SYSTEM_CONTROL, 'change_wallpaper', {'emotion': 'sad'}):
                self.set_wallpaper_for_emotion('sad')
            # Critical event: force wake up
            self.change_state(ElysiaState.AWAKE)

    def read_emotion_from_state_file(self):
        """Reads emotion from elysia_state.json and may change wallpaper."""
        try:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            state_path = os.path.join(project_root, 'elysia_state.json')
            if os.path.exists(state_path):
                with open(state_path, 'r', encoding='utf-8') as sf:
                    state = json.load(sf)
                    emotion = state.get('emotion', None)
                    if isinstance(emotion, str):
                        emotion_key = emotion.strip().lower()
                        if emotion_key and emotion_key != self.last_emotion:
                            if self.check_action_permission(ActionCategory.SYSTEM_CONTROL, 'change_wallpaper', {'emotion': emotion_key}):
                                self.set_wallpaper_for_emotion(emotion_key)
                                self.last_emotion = emotion_key
                            else:
                                self.logger.warning(f"Change wallpaper blocked for emotion={emotion_key}")
        except Exception as e:
            self.logger.error(f"Error reading state for wallpaper update: {e}")

    def trigger_learning(self):
        """
        Triggers the experience integration and memory weaving process during the IDLE state.
        This is Elysia's "dreaming" process.
        """
        self.logger.info("Dream cycle initiated. Weaving memories and integrating experiences...")

        # --- Part 1: Weave memories into insights ---
        try:
            self.memory_weaver.weave_memories()
        except Exception as e:
            self.logger.error(f"A critical error occurred during the memory weaving part of the dream cycle: {e}")

        # --- Part 2: Integrate raw experience logs ---
        # This part remains to process low-level logs into the CoreMemory knowledge base
        if not os.path.exists(self.experience_log_path):
            self.logger.info("No experience log found. Nothing to integrate.")
            return

        try:
            current_size = os.path.getsize(self.experience_log_path)
            if current_size <= self.last_experience_log_size:
                self.logger.info("No new experiences to integrate. Integration part of dream cycle peaceful.")
                if current_size < self.last_experience_log_size:
                    self.last_experience_log_size = current_size
                return

            with open(self.experience_log_path, 'r', encoding='utf-8') as f:
                f.seek(self.last_experience_log_size)
                new_experiences = f.readlines()
                integrated_count = 0
                for line in new_experiences:
                    if not line.strip(): continue
                    try:
                        log_entry = json.loads(line)
                        content = f"Event: {log_entry.get('type')}, Source: {log_entry.get('source')}, Data: {json.dumps(log_entry.get('data', {}))}"
                        self.experience_integrator.add_experience(content=content, category=log_entry.get('type', 'unknown'), context=log_entry.get('source', 'unknown'))
                        integrated_count += 1
                    except json.JSONDecodeError:
                        self.logger.warning(f"Could not parse line in experience log: {line.strip()}")
                    except Exception as e:
                        self.logger.error(f"Error integrating one experience: {e}")

                self.last_experience_log_size = f.tell()
                if integrated_count > 0:
                    self.logger.info(f"Dream cycle integration part completed. Integrated {integrated_count} new experiences.")
                    self.experience_integrator.save_memory()
                else:
                    self.logger.info("Finished integration part of dream cycle. No valid new experiences found.")
        except Exception as e:
            self.logger.error(f"A critical error occurred during the experience integration part of the dream cycle: {e}")

    def check_action_permission(self, category, action, details=None):
        """행동 허용 여부를 확인합니다."""
        return self.safety.check_action_permission(category, action, details)

    def enforce_infant_safety(self):
        """영아기 수준의 안전 규칙을 적용합니다."""
        if not self.check_action_permission(ActionCategory.FILE_ACCESS, "write", {"path": "/system/config.json"}):
            self.logger.warning("Prevented unauthorized file access")

    def enforce_toddler_safety(self):
        """유아기 수준의 안전 규칙을 적용합니다."""
        if not self.check_action_permission(ActionCategory.FILE_ACCESS, "read", {"path": "data/learning.txt"}):
            self.logger.info("Monitoring file access patterns")

    def set_wallpaper_for_emotion(self, emotion_key):
        """주어진 감정에 해당하는 이미지로 Windows 바탕화면을 변경합니다."""
        try:
            filename = self.wallpaper_map.get(emotion_key, None)
            if not filename:
                self.logger.warning(f"No wallpaper mapping for emotion: {emotion_key}")
                return

            img_path = os.path.join(self.faces_dir, filename)
            if not os.path.exists(img_path):
                self.logger.error(f"Wallpaper image not found: {img_path}")
                return

            SPI_SETDESKWALLPAPER = 20
            # For Windows, the flag 3 means update INI file and send change message
            ctypes.windll.user32.SystemParametersInfoW(SPI_SETDESKWALLPAPER, 0, img_path, 3)
            self.logger.info(f"Wallpaper set to {img_path} for emotion {emotion_key}")
            log_experience('guardian', 'action', {'event': 'set_wallpaper', 'emotion': emotion_key, 'path': img_path})

        except Exception as e:
            self.logger.error(f"Exception in set_wallpaper_for_emotion: {e}")

if __name__ == "__main__":
    guardian = Guardian()
    guardian.monitor_and_protect()
