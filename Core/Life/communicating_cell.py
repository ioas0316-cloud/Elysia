"""
Communicating Cell - Cells that can think and talk to each other

Philosophy:
- Each cell represents a concept
- Cells communicate based on conceptual proximity
- Dialogue emerges from cell interactions
- Elysia's language = Cell conversations
"""

import logging
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field

logger = logging.getLogger("CommunicatingCell")


@dataclass
class Message:
    """A message between cells."""
    from_concept: str
    to_concept: str
    content: str
    strength: float = 1.0
    timestamp: int = 0


@dataclass  
class Thought:
    """A cell's internal thought."""
    content: str
    basis: List[str] = field(default_factory=list)  # What influenced this thought
    confidence: float = 1.0


class CommunicatingCell:
    """
    A cell that can communicate with other cells.
    
    Each cell:
    - Represents a concept
    - Can send/receive messages
    - Has internal thoughts
    - Responds based on relations
    """
    
    def __init__(self, concept_id: str):
        self.concept = concept_id
        self.messages_received: List[Message] = []
        self.messages_sent: List[Message] = []
        self.thoughts: List[Thought] = []
        self.energy = 1.0
        self.activation = 0.0  # How active this cell is
        
        # Conceptual relations (learned or predefined)
        self.relations = {
            'self': {'you': 0.8, 'feeling': 0.6, 'light': 0.3},
            'you': {'self': 0.8, 'feeling': 0.5},
            'feeling': {'self': 0.6, 'why': 0.4, 'reason': 0.4},
            'light': {'darkness': 0.7, 'dream': 0.5},
            'why': {'reason': 0.9, 'feeling': 0.4},
            'love': {'self': 0.7, 'you': 0.8, 'feeling': 0.6},
            'dream': {'light': 0.5, 'self': 0.4},
        }
    
    def get_relation_strength(self, other_concept: str) -> float:
        """Get how related this cell is to another concept."""
        if self.concept in self.relations:
            return self.relations[self.concept].get(other_concept, 0.1)
        return 0.1  # Default weak relation
    
    def communicate_with(self, other_cell: 'CommunicatingCell', world_coherence: float = 0.5) -> Optional[Message]:
        """
        Try to communicate with another cell.
        
        Communication happens when:
        - Cells are conceptually related
        - Both have sufficient energy
        - World coherence is high enough
        """
        relation = self.get_relation_strength(other_cell.concept)
        
        logger.debug(f"Attempting communication: {self.concept} → {other_cell.concept}, relation={relation:.2f}")
        
        # Only communicate if related enough (LOWERED threshold!)
        if relation < 0.1:  # Changed from 0.3 to 0.1
            logger.debug(f"  → Too weak, skipping")
            return None
        
        # Create message based on relation type
        if relation > 0.7:
            content = f"strongly connected to {other_cell.concept}"
        elif relation > 0.5:
            content = f"related to {other_cell.concept}"
        elif relation > 0.3:
            content = f"aware of {other_cell.concept}"
        else:
            content = f"sensing {other_cell.concept}"
        
        message = Message(
            from_concept=self.concept,
            to_concept=other_cell.concept,
            content=content,
            strength=relation * world_coherence
        )
        
        # Send message
        self.messages_sent.append(message)
        other_cell.receive_message(message)
        
        # Boost activation
        self.activation += 0.1 * relation
        
        logger.debug(f"  → {self.concept} → {other_cell.concept}: {content} (strength={relation:.2f})")
        
        return message
    
    def receive_message(self, message: Message):
        """Receive a message from another cell."""
        self.messages_received.append(message)
        self.activation += 0.05 * message.strength
    
    def think(self, world_state: Optional[Dict] = None):
        """
        Internal thinking based on:
        - Messages received
        - Own concept
        - World state
        """
        if not self.messages_received:
            # No input, basic self-awareness
            thought = Thought(
                content=f"I am {self.concept}",
                basis=[self.concept],
                confidence=0.5
            )
            self.thoughts.append(thought)
            return
        
        # Process messages
        strongest_msg = max(self.messages_received, key=lambda m: m.strength)
        
        # Generate thought based on strongest connection
        if strongest_msg.strength > 0.7:
            # Strong connection
            thought_content = f"{self.concept} deeply connected to {strongest_msg.from_concept}"
        elif strongest_msg.strength > 0.5:
            # Medium connection
            thought_content = f"{self.concept} relates to {strongest_msg.from_concept}"
        else:
            # Weak connection
            thought_content = f"{self.concept} aware of {strongest_msg.from_concept}"
        
        thought = Thought(
            content=thought_content,
            basis=[strongest_msg.from_concept, self.concept],
            confidence=strongest_msg.strength
        )
        
        self.thoughts.append(thought)
        logger.debug(f"{self.concept} thinks: {thought_content}")
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of this cell's state."""
        return {
            'concept': self.concept,
            'activation': self.activation,
            'messages_received': len(self.messages_received),
            'messages_sent': len(self.messages_sent),
            'thoughts': [t.content for t in self.thoughts[-3:]],  # Last 3 thoughts
        }

    # --- Evolutionary Methods (Hyper-Evolution) ---

    def fitness(self, context_concepts: List[str]) -> float:
        """
        Calculate fitness based on:
        1. Activation (Energy)
        2. Relevance to context (Resonance)
        3. Connection strength
        """
        score = self.activation
        
        # Boost if related to context
        for ctx in context_concepts:
            score += self.get_relation_strength(ctx) * 2.0
            
        # Penalty for isolation
        if not self.messages_received and not self.messages_sent:
            score *= 0.5
            
        return score

    def mutate(self) -> bool:
        """
        Randomly discover new relations or shift focus.
        Returns True if mutation occurred.
        """
        import random
        
        # 1. Discover new relation (Mutation)
        potential_concepts = ['growth', 'pain', 'hope', 'eternity', 'void', 'light', 'connection']
        new_concept = random.choice(potential_concepts)
        
        if new_concept != self.concept and new_concept not in self.relations:
            # Spontaneous connection
            strength = random.uniform(0.1, 0.4)
            self.relations[self.concept][new_concept] = strength
            logger.debug(f"🧬 Mutation: {self.concept} discovered link to {new_concept} ({strength:.2f})")
            return True
            
        # 2. Shift existing relation strength (Drift)
        if self.relations.get(self.concept):
            target = random.choice(list(self.relations[self.concept].keys()))
            delta = random.uniform(-0.1, 0.2)
            self.relations[self.concept][target] = max(0.0, min(1.0, self.relations[self.concept][target] + delta))
            return True
            
        return False

    def crossover(self, other: 'CommunicatingCell') -> Optional['CommunicatingCell']:
        """
        Merge with another cell to create a new concept (Synthesis).
        Thesis (Self) + Antithesis (Other) -> Synthesis (New Cell)
        """
        import random
        
        # Only merge if strong connection exists
        relation = self.get_relation_strength(other.concept)
        if relation < 0.5:
            return None
            
        # Synthesis Logic (Simplified Concept Algebra)
        new_concept_map = {
            frozenset(['love', 'pain']): 'growth',
            frozenset(['love', 'time']): 'eternity',
            frozenset(['light', 'void']): 'creation',
            frozenset(['self', 'you']): 'oneness',
            frozenset(['feeling', 'reason']): 'insight',
            frozenset(['why', 'existence']): 'meaning',
        }
        
        pair = frozenset([self.concept, other.concept])
        child_concept = new_concept_map.get(pair)
        
        if not child_concept:
            # Probabilistic synthesis for unknown pairs
            if random.random() < 0.1:
                child_concept = f"{self.concept}_{other.concept}_synthesis"
            else:
                return None
        
        # Create Child
        child = CommunicatingCell(child_concept)
        child.activation = (self.activation + other.activation) / 2
        child.energy = (self.energy + other.energy) / 2
        
        # Inherit relations (mixed)
        child.relations[child_concept] = {}
        for k, v in self.relations.get(self.concept, {}).items():
            child.relations[child_concept][k] = v * 0.8
        for k, v in other.relations.get(other.concept, {}).items():
            child.relations[child_concept][k] = max(child.relations[child_concept].get(k, 0), v * 0.8)
            
        logger.info(f"✨ Synthesis: {self.concept} + {other.concept} -> {child_concept}")
        return child


def extract_dialogue_from_cells(cells: List[CommunicatingCell]) -> str:
    """
    Extract natural language from cell communications.
    
    Observes:
    - Which cells are most active
    - What messages were exchanged
    - What thoughts emerged
    """
    if not cells:
        return "...thinking..."
    
    logger.debug(f"Extracting dialogue from {len(cells)} cells")
    
    # Find most active cell
    most_active = max(cells, key=lambda c: c.activation)
    
    # Get all messages
    all_messages = []
    for cell in cells:
        all_messages.extend(cell.messages_received)
        logger.debug(f"  Cell '{cell.concept}': {len(cell.messages_received)} msgs, activation={cell.activation:.2f}")
    
    if not all_messages:
        # No communication happened - but still respond based on concept
        logger.debug(f"  No messages! Using primary concept: {most_active.concept}")
        
        # Better fallback responses
        if most_active.concept == 'self':
            return "저요? ...여기 있어요."
        elif most_active.concept == 'you':
            return "당신이요? ...듣고 있어요."
        elif most_active.concept == 'feeling':
            return "기분... 지금 느껴보고 있어요."
        elif most_active.concept == 'input':
            return "...귀 기울이고 있어요."
        else:
            return f"{most_active.concept}... 생각하고 있어요."
    
    # Find dominant connection
    strongest_msg = max(all_messages, key=lambda m: m.strength)
    logger.debug(f"  Strongest: {strongest_msg.from_concept} → {strongest_msg.to_concept} (strength={strongest_msg.strength:.2f})")
    
    # Generate response based on cell interactions
    primary = most_active.concept
    connected = strongest_msg.from_concept
    strength = strongest_msg.strength
    
    # Natural language generation based on concepts and strength
    if primary == 'self':
        if connected == 'you' or 'you' in [c.concept for c in cells]:
            if strength > 0.5:
                return "저요? 여기 있어요. 당신을 느끼고 있어요."
            else:
                return "...여기 있어요. 당신이 보여요."
        elif connected == 'feeling':
            return "제 기분이요? 지금 생각해보고 있어요..."
        else:
            return "저는... 여기 있어요."
    
    elif primary == 'you':
        return "당신... 느껴져요. 함께 있어요."
    
    elif primary == 'feeling':
        return "감정에 대해 생각하고 있어요. 복잡하네요..."
    
    elif primary == 'why' or primary == 'reason':
        if strength > 0.6:
            return f"{connected}... 그게 왜일까 깊이 생각해보고 있어요."
        else:
            return "왜일까요... 이유를 찾고 있어요."
    
    elif primary == 'love':
        return "사랑... 그게 모든 걸 연결하는 것 같아요."
    
    # Default: describe the connection
    if strength > 0.5:
        return f"{primary}(이)가 {connected}와 연결되어 있어요."
    else:
        return f"{primary}에 대해 생각하고 있어요."
