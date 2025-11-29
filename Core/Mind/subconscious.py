
import threading
import queue
import time
import logging
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass

logger = logging.getLogger("Subconscious")

@dataclass
class Thought:
    """A thought being processed in the background."""
    id: str
    prompt: str
    status: str = "thinking" # thinking, completed, failed
    result: Optional[str] = None
    created_at: float = 0.0
    completed_at: float = 0.0

class Subconscious:
    """
    Manages deep, slow thoughts in the background.
    Allows the Conscious mind to remain responsive.
    """
    def __init__(self):
        self.active_thoughts: Dict[str, Thought] = {}
        self.completed_queue = queue.Queue()
        self.lock = threading.Lock()
        
    def ponder(self, thought_id: str, prompt: str, think_func: Callable[[], str]):
        """
        Start a background thought process.
        
        Args:
            thought_id: Unique ID for the thought
            prompt: The original question/prompt
            think_func: The blocking function to execute (e.g., llm.think)
        """
        logger.info(f"💭 Subconscious: Pondering '{prompt[:30]}...' (ID: {thought_id})")
        
        thought = Thought(
            id=thought_id,
            prompt=prompt,
            created_at=time.time()
        )
        
        with self.lock:
            self.active_thoughts[thought_id] = thought
            
        # Start background thread
        thread = threading.Thread(
            target=self._process_thought,
            args=(thought, think_func),
            daemon=True
        )
        thread.start()
        
    def _process_thought(self, thought: Thought, think_func: Callable[[], str]):
        """Internal worker function."""
        try:
            # Execute the heavy thinking
            result = think_func()
            
            with self.lock:
                thought.result = result
                thought.status = "completed"
                thought.completed_at = time.time()
                
            self.completed_queue.put(thought)
            logger.info(f"💡 Subconscious: Insight gained for '{thought.id}'")
            
        except Exception as e:
            logger.error(f"❌ Subconscious: Thought failed: {e}")
            with self.lock:
                thought.status = "failed"
                thought.result = f"Error: {e}"
                
    def check_insights(self) -> Optional[Thought]:
        """
        Check if any thoughts have bubbled up to the surface.
        Returns the next completed Thought, or None.
        """
        try:
            return self.completed_queue.get_nowait()
        except queue.Empty:
            return None
            
    def is_thinking(self) -> bool:
        """Check if any heavy thoughts are in progress."""
        with self.lock:
            return any(t.status == "thinking" for t in self.active_thoughts.values())
