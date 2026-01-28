from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, List, Optional

from Core.L5_Mental.Memory.core_memory import CoreMemory
from Core.L5_Mental.Reasoning_Core.HighEngine.quaternion_engine import QuaternionConsciousnessEngine, QuaternionOrientation
from Core.L5_Mental.Reasoning_Core.HighEngine.syllabic_language_engine import SyllabicLanguageEngine


def _analyze_user_input(text: str) -> Dict[str, str]:
    """
    Simplified auditory cortex: deduce intent/emotion from text features.
    """
    cleaned = text.strip()
    length = len(cleaned)
    intent = {"intent_type": "unknown", "emotion": "neutral"}
    if "?" in cleaned or " " in cleaned or " " in cleaned:
        intent["intent_type"] = "reflect"
        intent["emotion"] = "curious"
    elif "!" in cleaned:
        intent["intent_type"] = "act"
        intent["emotion"] = "joy"
    elif "  " in cleaned or "  " in cleaned or "  " in cleaned:
        intent["intent_type"] = "dream"
        intent["emotion"] = "happy"
    elif "  " in cleaned or "  " in cleaned:
        intent["intent_type"] = "respond"
        intent["emotion"] = "sad"
    elif length < 5:
        intent["intent_type"] = "respond"
        intent["emotion"] = "neutral"
    else:
        intent["intent_type"] = "propose_action"
        intent["emotion"] = "relief"
    return intent


class ToddlerChatEngine:
    """
    Encapsulates the quaternion   language   emotion loop and streams logs.
    """

    MOOD_LOG = "elysia_logs/soul_state.jsonl"
    MOOD_MAP = {
        "dream": "calm",
        "act": "curious",
        "reflect": "worry",
        "respond": "joy",
        "unknown": "pout",
        "propose_action": "hopeful",
    }

    def __init__(self, memory_path: str = "data/Memory/elysia_core_memory.json") -> None:
        log_dir = os.path.dirname(self.MOOD_LOG)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        self.log_path = self.MOOD_LOG
        self.memory = CoreMemory(file_path=memory_path) if os.path.exists(memory_path) else CoreMemory(file_path=None)
        self.q_engine = QuaternionConsciousnessEngine(core_memory=self.memory)
        self.lang_engine = SyllabicLanguageEngine(core_memory=self.memory)

    def _gather_context(self, status: Dict[str, Any]) -> Dict[str, Any]:
        fragments = self.memory.get_identity_fragments(n=3)
        experiences = self.memory.get_experiences(n=3)
        identity_texts = [frag.content for frag in fragments if getattr(frag, "content", None)]
        experience_snippet = experiences[-1].content if experiences else ""
        return {
            "identity": identity_texts,
            "experience": experience_snippet,
            "status": status,
        }

    def _compose_thought_trail(self, intent: Dict[str, str], status: Dict[str, Any], context: Dict[str, Any]) -> str:
        trail: List[str] = []
        axis_focus = status.get("primary_focus", "Unknown")
        anchor = status.get("anchor_strength", 0.0)
        if anchor > 0.75:
            trail.append(f"        {axis_focus}                  .")
        else:
            trail.append(f"        {axis_focus}                  .")

        identity = context.get("identity")
        if identity:
            trail.append(f"       {identity[0][:30]}...          .")

        experience = context.get("experience")
        if experience:
            snippet = experience[:40]
            trail.append(f"      '{snippet}'         .")

        if intent.get("intent_type") == "dream":
            trail.append("                            .")

        return " ".join(trail).strip()

    def _compose_meta_observation(self, status: Dict[str, Any], context: Dict[str, Any]) -> str:
        focus = status.get("primary_focus", "Unknown")
        anchor = status.get("anchor_strength", 0.0)
        fragments = context.get("identity", [])
        obs = []
        if anchor < 0.4:
            obs.append("                          .")
        else:
            obs.append("                        .")

        if fragments:
            obs.append(f"          ('{fragments[0][:20]}...')                     .")
        obs.append(f"         '{focus}'   ,                            .")
        return " ".join(obs)

    def _map_visual_to_vector(
        self,
        description: str,
        palette: Optional[List[str]],
        brightness: float,
    ) -> Dict[str, float]:
        vector = {"w": 0.0, "x": 0.0, "y": 0.0, "z": 0.0}
        desc = description.lower()
        if " " in desc or " " in desc or " " in desc:
            vector["w"] += brightness * 0.4
            vector["z"] += 0.3
        if "  " in desc or "  " in desc or "  " in desc:
            vector["y"] += brightness * 0.4
        if "  " in desc or "  " in desc:
            vector["x"] += brightness * 0.2
        if palette:
            for color in palette:
                if "  " in color or "  " in color:
                    vector["y"] += 0.2
                if "  " in color or "  " in color:
                    vector["z"] += 0.2
        return vector

    def _determine_focus_area(self, orientation: QuaternionOrientation) -> str:
        axis_focus = max(
            (("w", orientation.w), ("x", orientation.x), ("y", orientation.y), ("z", orientation.z)),
            key=lambda pair: abs(pair[1]),
        )[0]
        focus_names = {
            "w": "      ",
            "x": "      ",
            "y": "       ",
            "z": "     ",
        }
        return focus_names.get(axis_focus, "     ")

    def _blend_orientation(self, add: Dict[str, float]) -> QuaternionOrientation:
        current = self.q_engine.orientation
        blended = QuaternionOrientation(
            w=current.w + add.get("w", 0.0),
            x=current.x + add.get("x", 0.0),
            y=current.y + add.get("y", 0.0),
            z=current.z + add.get("z", 0.0),
        )
        return blended.normalized()

    def process_visual_input(
        self,
        description: str,
        palette: Optional[List[str]] = None,
        brightness: float = 0.5,
    ) -> Dict[str, Any]:
        """
        Introduce a vision cue: adjust orientation, log, and surface a thought hint.
        """
        orientation_add = self._map_visual_to_vector(description, palette, brightness)
        target = self._blend_orientation(orientation_add)
        self.q_engine._orientation = self.q_engine._slerp(self.q_engine.orientation, target, alpha=0.4)

        status = self.q_engine.get_lens_status()
        context = self._gather_context(status)
        trail = self._compose_thought_trail({"intent_type": "dream"}, status=status, context=context)

        focus_area = self._determine_focus_area(target)
        note = f"{focus_area}          , '{description[:40]}'           ."
        log_entry = {
            "timestamp": time.time(),
            "mood": "calm",
            "speech": note,
            "thought_trail": trail,
            "meta_observation": self._compose_meta_observation(status, context),
            "vision_description": description,
            "vision_palette": palette or [],
            "focus_area": focus_area,
            "primary_focus": status["primary_focus"],
            "anchor_strength": status["anchor_strength"],
            "intent": "vision",
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        return {
            "mood": "calm",
            "speech": note,
            "thought_trail": trail,
            "status": status,
            "intent": "vision",
            "vision_description": description,
            "vision_palette": palette or [],
        }

    def process_input(self, text: str) -> Dict[str, Any]:
        """
        Process incoming text, update engines, log soulful output, and return response metadata.
        """
        if not text or not text.strip():
            return {}

        intent_bundle = _analyze_user_input(text)
        mood_hint = intent_bundle["emotion"]

        law_alignment = {}
        if intent_bundle["intent_type"] == "reflect":
            law_alignment = {"scores": {"truth": 0.5}}
        elif intent_bundle["intent_type"] == "act":
            law_alignment = {"scores": {"liberation": 0.5}}

        self.q_engine.update_from_turn(law_alignment=law_alignment, intent_bundle=intent_bundle)
        status = self.q_engine.get_lens_status()
        context = self._gather_context(status)
        thought = self._compose_thought_trail(intent_bundle, status, context)

        response_word = self.lang_engine.suggest_word(
            intent_bundle=intent_bundle,
            orientation=self.q_engine.orientation_as_dict(),
        )

        current_mood = self.MOOD_MAP.get(intent_bundle["intent_type"], "neutral")
        meta_obs = self._compose_meta_observation(status, context)
        log_entry = {
            "timestamp": time.time(),
            "mood": current_mood,
            "speech": response_word,
            "thought_trail": thought,
            "meta_observation": meta_obs,
            "user_input": text,
            "primary_focus": status["primary_focus"],
            "anchor_strength": status["anchor_strength"],
            "intent": intent_bundle["intent_type"],
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

        return {
            "mood": current_mood,
            "speech": response_word,
            "thought_trail": thought,
            "meta_observation": meta_obs,
            "status": status,
            "intent": intent_bundle["intent_type"],
            "emotion": mood_hint,
        }
