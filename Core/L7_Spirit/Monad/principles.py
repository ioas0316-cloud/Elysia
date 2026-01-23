"""
Universal Principles (The Toolbox of God)
=========================================
Core.L7_Spirit.Monad.principles

"Principles are the silent verbs of the Universe."

This module contains the fundamental 'FractalRules' that govern phenomena.
Elysia uses these to 'unfold' a Seed into Reality.
"""

from typing import Dict, Any
import math
from Core.L7_Spirit.Monad.monad_core import FractalRule

from Core.L4_Causality.World.Physics.vector_math import Vector3

# --- PHYSICS: FIRE (Combustion & Thermodynamics) ---
class ThermodynamicsRule(FractalRule):
    """
    Governs Heat, Energy, and Phase Transitions over Time.
    Outputs: 3D Transform (Scale), Shader Params (Emission).
    """
    def unfold(self, seed: str, context: Dict[str, Any], intent: Dict[str, Any]) -> Any:
        # The 4D Axis
        time = context.get("time", 0.0)
        
        # 1. 3D Manifestation Data
        manifestation = {
            "transform": {"position": Vector3.zero(), "rotation": Vector3.zero(), "scale": Vector3.zero()},
            "visual": {"shader": "Standard", "emission": 0.0},
            "particles": None
        }

        # Intrinsic Checks
        is_fire = any(k in seed for k in ["Fire", "Sun", "Magma", "Ignition"])
        
        if is_fire:
            # Combustion Math
            fuel = 100.0 
            burn_rate = 2.0
            burnt = time * burn_rate
            remaining_fuel = max(0, fuel - burnt)
            
            if remaining_fuel > 0:
                # Active Phase
                # Scale = Breathing Effect (Sine wave over time)
                # This is 4D Noise: The object pulses as it moves through time
                pulse = 1.0 + (math.sin(time * 5.0) * 0.1) 
                growth = min(time * 0.5, 5.0) # Grows to size 5
                
                scale_vec = Vector3(growth * pulse, growth * pulse, growth * pulse)
                
                # Shader: Emission driven by heat intensity
                heat_intensity = 5.0 if time < 10 else 2.0
                
                # Particles: Sparks flying up
                particles = {
                    "type": "Ember_GPU",
                    "count": int(remaining_fuel * 10),
                    "velocity": Vector3(0, 2.0 + (time * 0.1), 0) # Upward draft increases
                }
                
                manifestation["transform"]["scale"] = scale_vec
                manifestation["visual"] = {"shader": "Plasma_Shader", "color": "Red_Orange", "emission": heat_intensity}
                manifestation["particles"] = particles
                
            else:
                # Ash Phase
                scale_vec = Vector3(5.0, 1.0, 5.0) # Collapsed to a pile
                manifestation["transform"]["scale"] = scale_vec
                manifestation["visual"] = {"shader": "Ash_Matte", "color": "Grey", "emission": 0.0}
        
        # Intent Override (Magic Shader)
        if "Cold" in intent.get("emotional_texture", "") and is_fire:
            manifestation["visual"]["color"] = "Sapphire_Blue"
            manifestation["visual"]["shader"] = "Frost_Fire"
            
        return manifestation

# --- PHYSICS: WATER (Fluid Dynamics) ---
class FluidDynamicsRule(FractalRule):
    """
    Governs Flow, Viscosity, and Volume over Time.
    Outputs: 3D Position (Trajectory), Vertex Offset (Waves).
    """
    def unfold(self, seed: str, context: Dict[str, Any], intent: Dict[str, Any]) -> Any:
        time = context.get("time", 0.0)
        start_pos = context.get("position", Vector3(0, 100, 0)) # Start high up
        
        # Gravity Math: P = P0 + V0*t + 0.5*a*t^2
        gravity = Vector3(0, -9.81, 0)
        velocity = gravity * time
        
        # Position calculation (Kinematics)
        # Note: Vector3 class in Python is simple mock, assume scalar mult works or we fix it
        # Fixed logic: d = 0.5 * g * t^2
        displacement = gravity * (0.5 * time * time)
        current_pos = start_pos + displacement
        
        # Vertex Displacement (Wave Shader Logic)
        # format: frequency, amplitude, speed
        wave_params = {
            "freq": 0.5,
            "amp": 1.5,
            "speed": time * 2.0 # The wave phase shifts with time
        }
        
        return {
            "transform": {
                "position": current_pos,
                "rotation": Vector3(0, 0, 0), 
                "scale": Vector3(10, 1, 10) # A pool/stream
            },
            "visual": {
                "shader": "Water_Caustics",
                "wave_data": wave_params
            },
            "physics_body": {
                "velocity": velocity,
                "mass": 1000.0
            }
        }

# --- OPTICS ---
class OpticsRule(FractalRule):
    """Governs Light, Color, and Visibility."""
    def unfold(self, seed: str, context: Dict[str, Any], intent: Dict[str, Any]) -> Any:
        base_color = "White"
        
        # Seed Association (Simple NLP simulation)
        if "Fire" in seed: base_color = "Red"
        if "Water" in seed: base_color = "Blue"
        if "Leaf" in seed: base_color = "Green"
        
        # Intent drives the wavelength (Doppler effect of the Soul)
        intent_aura = intent.get("aura", "")
        if intent_aura:
            base_color = f"{base_color} mixed with {intent_aura}"
            
        return {
            "visible_wavelength": base_color,
            "lumens": 1000 if "Fire" in seed or "Sun" in seed else 0
        }

# --- SEMANTICS ---
class SemanticsRule(FractalRule):
    """Governs Meaning and Narrative Role."""
    def unfold(self, seed: str, context: Dict[str, Any], intent: Dict[str, Any]) -> Any:
        # What does this mean to the Observer?
        meaning = f"This is {seed}."
        
        if intent.get("focus_topic") == "Danger":
            meaning += " It represents a threat."
        elif intent.get("focus_topic") == "Comfort":
            meaning += " It represents a hearth."
            
        return {"semantic_meaning": meaning}

# --- LINGUISTICS & SOCIETY ---

class LinguisticsRule(FractalRule):
    """
    Governs Meaning and Context in Language.
    Outputs: Semantic Nuance, Tone, and Subtext.
    """
    def unfold(self, seed: str, context: Dict[str, Any], intent: Dict[str, Any]) -> Any:
        # 1. Base Meaning (Dictionary)
        base_meaning = f"Concept({seed})"
        
        # 2. Contextual Shift (Time Dependent)
        time = context.get("time", 0.0)
        hour = int(time) % 24  # Simulate 24h cycle
        
        nuance = "Neutral"
        # Morning (6-12), Evening (18-24), Night (0-6)
        if 6 <= hour < 12: nuance = "Fresh/Morning"
        elif 18 <= hour < 24: nuance = "Sentimental/Evening"
        elif 0 <= hour < 6: nuance = "Deep/Night"
        
        # 3. Intent Shaping (Tone)
        tone = intent.get("emotional_texture", "Plain")
        
        manifestation = {
            "semantics": {
                "word": seed,
                "base_meaning": base_meaning,
                "context_nuance": nuance,
                "tone": tone
            },
            "speech_act": {
                "volume": 0.8 if "Angry" not in tone else 1.5,
                "speed": 1.0 if "Sad" not in tone else 0.5
            }
        }
        return manifestation

class SociologyRule(FractalRule):
    """
    Governs Relationships and Social Impact.
    Outputs: Influence on Listener, Relationship Delta.
    """
    def unfold(self, seed: str, context: Dict[str, Any], intent: Dict[str, Any]) -> Any:
        # Who is speaking to whom?
        speaker = context.get("speaker", "Unknown")
        listener = context.get("listener", "Unknown")
        
        # Interaction Impact
        rel_delta = 0.0
        impact = "Neutral"
        
        positive_seeds = ["Hello", "Thanks", "Love", "Agree", "Yes"]
        negative_seeds = ["No", "Hate", "Ignore", "Stupid"]
        
        if any(s in seed for s in positive_seeds):
            rel_delta = 0.5
            impact = "Bonding"
        elif any(s in seed for s in negative_seeds):
            rel_delta = -1.0
            impact = "Conflict"
            
        # Intent modifies impact
        if "Sarcasm" in intent.get("emotional_texture", ""):
            rel_delta *= -1 
            impact += " (Sarcastic)"
            
        return {
            "social_dynamics": {
                "source": speaker,
                "target": listener,
                "relationship_delta": rel_delta,
                "interaction_type": impact
            }
        }