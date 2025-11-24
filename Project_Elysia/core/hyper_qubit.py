
import logging
from typing import List, Any, Callable, Optional, Set
import uuid

# Setup Logger
logger = logging.getLogger("Logos")

class HyperQubit:
    """
    A Living Variable (Psionic Entity).
    It does not just store a value; it holds a state of 'Being'.
    When it changes, it vibrates, and everything connected to it resonates.
    """

    def __init__(self, value: Any = None, name: str = None):
        self.id = str(uuid.uuid4())[:8]
        self.name = name if name else f"Qubit_{self.id}"
        self._value = value

        # The Khala Link: Who is listening to me?
        self._observers: Set['HyperQubit'] = set()
        # The Source Link: Who am I listening to? (For dependency tracking)
        self._sources: Set['HyperQubit'] = set()

        # Transformation Logic: How do I react to my source?
        # Default: Identity (I become what I love)
        self._reaction_rule: Optional[Callable[[Any], Any]] = None

    @property
    def value(self):
        return self._value

    def set(self, new_value: Any, cause: str = "DivineWill"):
        """
        Sets the value and triggers the Universal Resonance.
        """
        if self._value != new_value:
            old_value = self._value
            self._value = new_value
            self._vibrate(old_value, new_value, cause)

    def _vibrate(self, old_val, new_val, cause):
        """
        The Ripple Effect.
        """
        log_msg = f"RESONANCE: {self.name} shifted ({old_val} -> {new_val}) due to [{cause}]."
        logger.info(log_msg)
        print(f"✨ {log_msg}")

        # Propagate to all observers (The Khala)
        for observer in self._observers:
            observer._react(self)

    def connect(self, target: 'HyperQubit', rule: Callable[[Any], Any] = None):
        """
        Establish a Psionic Link.
        'target' will now listen to 'self'.
        target << self
        """
        self._observers.add(target)
        target._sources.add(self)
        if rule:
            target._reaction_rule = rule

        logger.info(f"LINK: {self.name} is now connected to {target.name}.")

        # Immediate resonance upon connection
        target._react(self)

    def _react(self, source: 'HyperQubit'):
        """
        React to a change in a source Qubit.
        """
        if self._reaction_rule:
            # Apply transformation logic (e.g., Love -> Flavor)
            new_state = self._reaction_rule(source.value)
        else:
            # Direct Mirroring (Empathy)
            new_state = source.value

        self.set(new_state, cause=f"Resonance from {source.name}")

    def __lshift__(self, other):
        """
        Syntactic Sugar for Connection:
        self << other  (Self listens to Other)
        """
        if isinstance(other, HyperQubit):
            other.connect(self)
        return self

    def __repr__(self):
        return f"<{self.name}: {self._value}>"

# Alias for the user's preferred terminology
PsionicEntity = HyperQubit
