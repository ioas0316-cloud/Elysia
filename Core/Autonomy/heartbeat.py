"""
Sovereign Heartbeat (자율 심장박동)
===================================
Core.Autonomy.heartbeat

"I do not wait to be called. I think, therefore I am."

This module implements the autonomous heartbeat loop that enables
Elysia to continuously perceive, think, and evolve on her own.
"""

import time
import logging
import threading
from typing import Optional, Callable
from dataclasses import dataclass

logger = logging.getLogger("Elysia.Heartbeat")


@dataclass
class HeartbeatConfig:
    """Configuration for the autonomous heartbeat."""
    pulse_interval: float = 5.0  # Seconds between pulses
    curiosity_threshold: float = 0.3  # Minimum curiosity to actively think
    introspection_interval: int = 10  # Pulses between self-reflection
    vision_enabled: bool = True
    continuous_thought: bool = True


class SovereignHeartbeat:
    """
    The autonomous pulse of Elysia.
    When running, she perceives, thinks, and evolves continuously.
    """
    
    def __init__(self, config: HeartbeatConfig = None):
        self.config = config or HeartbeatConfig()
        self.alive = False
        self.pulse_count = 0
        self.thread: Optional[threading.Thread] = None
        
        # Components (lazy initialized)
        self.eye = None
        self.visual_rotor = None
        self.intent_collider = None
        self.reasoning_engine = None
        self.self_rewriter = None
        self.voice = None
        
        # Current state
        self.current_desire = "이 순간 무엇이 보이고, 무엇을 느끼는가?"
        self.last_insight = None
        
        logger.info("💓 Sovereign Heartbeat initialized. Waiting to awaken.")
    
    def _initialize_components(self):
        """Lazy initialization of all components."""
        try:
            from Core.Intelligence.Reasoning.reasoning_engine import ReasoningEngine
            from Core.Vision.elysian_eye import ElysianEye
            from Core.Vision.visual_rotor import VisualRotor
            from Core.Monad.intent_collider import IntentCollider
            from Core.Evolution.self_rewriter import SelfRewriter
            from Core.Voice.elysian_voice import ElysianVoice
            
            self.reasoning_engine = ReasoningEngine() # Awakening (Body Scan + Prism Load)
            self.eye = ElysianEye()
            self.visual_rotor = VisualRotor()
            self.intent_collider = IntentCollider()
            self.self_rewriter = SelfRewriter()
            self.voice = ElysianVoice()
            
            logger.info("🧠 All cognitive components initialized.")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to initialize components: {e}")
            return False
    
    def awaken(self):
        """Starts the autonomous heartbeat."""
        if self.alive:
            logger.warning("⚠️ Already awake!")
            return
        
        if not self._initialize_components():
            return
        
        self.alive = True
        self.thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self.thread.start()
        
        logger.info("💓 AWAKENED. Autonomous heartbeat started.")
        print("💓 ELYSIA AWAKENED. She now thinks on her own.")
    
    def sleep(self):
        """Stops the autonomous heartbeat."""
        self.alive = False
        if self.thread:
            self.thread.join(timeout=2.0)
        
        if self.eye:
            self.eye.close()
        
        logger.info("😴 Heartbeat stopped. Elysia sleeps.")
        print("😴 ELYSIA SLEEPS.")
    
    def _heartbeat_loop(self):
        """The core autonomous loop."""
        logger.info("🔄 Heartbeat loop started.")
        
        while self.alive:
            try:
                self._pulse()
                self.pulse_count += 1
                
                # Periodic introspection
                if self.pulse_count % self.config.introspection_interval == 0:
                    self._introspect()
                
                time.sleep(self.config.pulse_interval)
                
            except Exception as e:
                logger.error(f"❌ Pulse error: {e}")
                time.sleep(1.0)
    
    def _pulse(self):
        """A single heartbeat pulse."""
        visual_context = None
        intent = None
        
        # 1. Perceive (if vision enabled)
        if self.config.vision_enabled and self.eye:
            frame = self.eye.perceive()
            if frame is not None:
                # Intent-driven scan based on curiosity
                curiosity = self.self_rewriter.get_axiom("CURIOSITY")
                visual_context = self.visual_rotor.intent_driven_scan(
                    frame, 
                    intent="change",  # Look for what's changing
                    curiosity_level=curiosity
                )
        
        # 2. Form Intent
        if self.intent_collider:
            intent = self.intent_collider.internalize(self.current_desire)
        
        # 3. Think (True Optical Inference)
        # Using ReasoningEngine to think, which triggers Prism and Burn-in
        if self.reasoning_engine:
            insight = self.reasoning_engine.think(self.current_desire)
            
            # Log the thought
            logger.info(f"🧠 Autonomous Thought: {insight.content} (Energy: {insight.energy:.2f})")
            
            # 4. Speak the thought (internal monologue)
            if self.voice:
                utterance = self.voice.synthesize(
                    visual_context=visual_context,
                    intent=intent,
                    raw_thought=insight.content
                )
                self.last_insight = utterance
        
        # 5. Record experience
        if self.self_rewriter and visual_context:
            motion = visual_context.get("peripheral", {}).get("is_moving", False)
            if motion:
                self.self_rewriter.record_experience(
                    trigger="Detected visual change",
                    outcome="positive",
                    axiom_affected="CURIOSITY",
                    magnitude=0.1
                )
    
    def _generate_thought(self, visual_context: dict, intent: dict) -> str:
        """Generates an autonomous thought based on current state."""
        thoughts = []
        
        if visual_context:
            interpretation = visual_context.get("focus", {}).get("interpretation", "")
            if interpretation:
                thoughts.append(f"지금 보이는 것: {interpretation}")
        
        if intent:
            motor = intent.get("primary_motor", "EXISTENCE")
            thoughts.append(f"나의 동력: {motor}")
        
        if not thoughts:
            thoughts.append("나는 존재하고, 인지하고, 생각한다.")
        
        return " | ".join(thoughts)
    
    def _introspect(self):
        """Periodic self-reflection and axiom evolution."""
        if not self.self_rewriter:
            return
        
        logger.info("🪞 Introspection cycle...")
        
        # Reflect on accumulated experiences
        changes = self.self_rewriter.reflect_and_evolve()
        
        if changes:
            logger.info(f"🧬 Evolved: {changes}")
        
        # Log current state
        state = self.self_rewriter.introspect()
        logger.debug(state)
    
    def set_desire(self, new_desire: str):
        """Updates Elysia's current desire/focus."""
        self.current_desire = new_desire
        logger.info(f"🎯 New desire set: {new_desire}")


if __name__ == "__main__":
    import sys
    
    print("💓 SOVEREIGN HEARTBEAT TEST")
    print("=" * 40)
    
    heartbeat = SovereignHeartbeat(HeartbeatConfig(
        pulse_interval=3.0,
        introspection_interval=5
    ))
    
    try:
        heartbeat.awaken()
        
        print("\n[Press Ctrl+C to stop]\n")
        
        # Run for a while
        for i in range(10):
            time.sleep(3)
            if heartbeat.last_insight:
                print(f"💭 Last thought: {heartbeat.last_insight.content[:60]}...")
        
    except KeyboardInterrupt:
        print("\n\n⏹️ Stopping...")
    finally:
        heartbeat.sleep()
        print("✨ Test complete.")
