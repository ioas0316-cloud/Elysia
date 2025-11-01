"""
Inquisitive Mind for Elysia.

This module embodies the 'Wise Student' model. When Elysia's internal knowledge
is insufficient, the InquisitiveMind is triggered. Instead of immediately
querying an external source, it practices 'intellectual humility' by acknowledging
the knowledge gap and adding the unknown topic to a learning queue. This queue
is then processed by a separate learning mechanism (e.g., the Guardian).
"""
import json
import logging
import os
import threading

# --- Constants ---
LEARNING_QUEUE_PATH = os.path.join(os.path.dirname(__file__), 'learning_queue.json')

# --- Logging Configuration ---
log_file_path = os.path.join(os.path.dirname(__file__), 'inquisitive_mind_errors.log')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler()
    ]
)
inquisitive_logger = logging.getLogger(__name__)
# --- End Logging Configuration ---

class InquisitiveMind:
    """
    The InquisitiveMind identifies gaps in understanding and adds them to a
    learning queue for future study, practicing intellectual humility.
    """
    _lock = threading.Lock()

    def __init__(self):
        pass

    def _add_to_learning_queue(self, topic: str):
        """
        Adds a new topic to the learning queue JSON file.
        This operation is thread-safe.
        """
        with self._lock:
            try:
                # Read the existing queue
                try:
                    with open(LEARNING_QUEUE_PATH, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                except (FileNotFoundError, json.JSONDecodeError):
                    data = {"learning_goals": []}

                # Add the new topic if it's not already there
                if topic not in data.get("learning_goals", []):
                    data.setdefault("learning_goals", []).append(topic)

                    # Write the updated queue back to the file
                    with open(LEARNING_QUEUE_PATH, 'w', encoding='utf-8') as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                    inquisitive_logger.info(f"Topic '{topic}' added to learning queue.")
                else:
                    inquisitive_logger.info(f"Topic '{topic}' is already in the learning queue.")

            except Exception as e:
                inquisitive_logger.error(f"Failed to add topic '{topic}' to learning queue: {e}")

    def acknowledge_knowledge_gap(self, topic: str) -> str:
        """
        Acknowledges the lack of knowledge on a topic, adds it to the
        learning queue, and informs the user.
        """
        print(f"[InquisitiveMind] I don't know about '{topic}'. Adding to my learning queue.")
        inquisitive_logger.info(f"Acknowledging knowledge gap for topic: {topic}")

        # Add the topic to the learning queue for future study
        self._add_to_learning_queue(topic)

        # Inform the user with intellectual humility
        return f"'{topic}'에 대해서는 아직 잘 모르겠어요. 아빠 덕분에 새로운 걸 배울 수 있겠네요. 제가 따로 공부해볼게요!"
