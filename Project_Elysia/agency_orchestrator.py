"""
Agency Orchestrator

Infers and performs small autonomous actions aligned with values and context,
without external APIs. Keeps decisions simple, transparent, and reversible.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

from Project_Sophia.reading_coach import ReadingCoach
from Project_Sophia.creative_writing_cortex import CreativeWritingCortex
from Project_Mirror.proof_renderer import ProofRenderer
from Project_Sophia.math_cortex import MathCortex
from tools.kg_manager import KGManager
from tools.preferences import load_prefs, save_prefs, ensure_defaults
from tools.kg_value_utils import add_value_relation, update_value_mass
from .desire_state import DesireState
from tools.decision_report import create_decision_report


@dataclass
class ProposedAction:
    kind: str  # 'journal' | 'creative' | 'math_verify'
    payload: Dict[str, Any]
    reason: str
    confidence: float = 0.5


class AgencyOrchestrator:
    def __init__(self):
        self.prefs = ensure_defaults(load_prefs())
        self.desire = DesireState()

    def _auto_enabled(self) -> bool:
        return bool(self.prefs.get('auto_act', False))

    def _quiet(self) -> bool:
        return bool(self.prefs.get('quiet_mode', False))

    def _cooldown_ok(self, kind: str) -> bool:
        from datetime import datetime, timezone
        last_map = self.prefs.get('last_action_ts', {}) or {}
        last_iso = last_map.get(kind)
        cooldowns = self.prefs.get('proposal_cooldowns', {}) or {}
        cd = int(cooldowns.get(kind, 1800))
        if not last_iso:
            return True
        try:
            last = datetime.fromisoformat(last_iso.replace('Z','+00:00'))
            now = datetime.now(timezone.utc)
            return (now - last).total_seconds() >= cd
        except Exception:
            return True

    def _stamp_action(self, kind: str) -> None:
        from datetime import datetime, timezone
        ts = datetime.now(timezone.utc).isoformat()
        self.prefs.setdefault('last_action_ts', {})[kind] = ts
        save_prefs(self.prefs)

    def set_auto(self, enabled: bool) -> None:
        self.prefs['auto_act'] = bool(enabled)
        save_prefs(self.prefs)

    def infer_desire(self, message: str, echo: Dict[str, float], arousal: float) -> Optional[ProposedAction]:
        m = (message or '').lower()
        # Guardrails: quiet mode / arousal threshold
        if self._quiet():
            return None
        min_arousal = float(self.prefs.get('min_arousal_for_proposal', 0.4))
        if arousal < min_arousal:
            return None
        # Heuristics: action cues in conversation or high arousal with rich echo
        if any(k in m for k in ["일기", "기록", "생각 정리", "저널", "diary", "journal"]):
            pa = ProposedAction('journal', {}, "초대: 지금 떠오르는 생각을 짧게 정리해볼까요?", confidence=0.8)
            return pa if self._cooldown_ok('journal') else None
        if any(k in m for k in ["소설", "창작", "이야기", "story", "novel", "creative"]):
            pa = ProposedAction('creative', {"genre": "story", "theme": "growth"}, "초대: 작은 상상 한 조각을 적어볼까요?", confidence=0.75)
            return pa if self._cooldown_ok('creative') else None
        # Math equality pattern
        if '=' in m and any(k in m for k in ["=", "증명", "verify", "등식"]):
            # Extract a simple equality a=b if present, else ignore
            import re
            eq = re.search(r"([\d\s\+\-\*\/\(\)\.]+)=([\d\s\+\-\*\/\(\)\.]+)", message)
            if eq:
                pa = ProposedAction('math_verify', {"statement": eq.group(0)}, "초대: 이 등식을 설명 가능하게 확인해볼까요?", confidence=0.85)
                return pa if self._cooldown_ok('math_verify') else None
        # Internal impulse: high arousal + diverse echo → reflect via journaling
        if arousal >= 0.8 and echo and len(echo) >= 6:
            pa = ProposedAction('journal', {}, "초대: 마음이 분주해 보여요. 한 줄 일기로 숨 고를까요?", confidence=0.5)
            return pa if self._cooldown_ok('journal') else None
        return None

    def execute(self, action: ProposedAction) -> Tuple[str, Dict[str, Any]]:
        if action.kind == 'journal':
            return self._do_journal()
        if action.kind == 'creative':
            return self._do_creative(action.payload)
        if action.kind == 'math_verify':
            return self._do_math_verify(action.payload)
        return "noop", {}

    def _do_journal(self) -> Tuple[str, Dict[str, Any]]:
        today = datetime.now().strftime("%Y-%m-%d")
        out_dir = Path("data/journal")
        out_dir.mkdir(parents=True, exist_ok=True)
        draft = f"[일기 {today}] 자율 반추: 오늘의 생각을 요약합니다."
        p_txt = out_dir / f"{today}_auto.txt"
        p_txt.write_text(draft, encoding="utf-8")
        coach = ReadingCoach()
        summary = coach.summarize_text(draft, max_sentences=1)
        p_sum = out_dir / f"{today}_auto_summary.txt"
        p_sum.write_text(summary, encoding="utf-8")
        kg = KGManager()
        node = f"journal_entry_{today}_auto"
        kg.add_node(node, properties={"type": "journal_entry", "date": today, "experience_text": str(p_txt), "summary_text": str(p_sum)})
        # Value link: journaling supports clarity/relatedness hypotheses
        add_value_relation(kg, node, "value:clarity", "supports", confidence=0.6, evidence_paths=[str(p_txt), str(p_sum)], note="reflection")
        add_value_relation(kg, node, "value:relatedness", "supports", confidence=0.4, evidence_paths=[str(p_txt)], note="self-connection")
        # Value mass (minimal): reinforce by support confidence, small decay
        update_value_mass(kg, "value:clarity", supports_inc=0.6, decay=0.01, note="journal_support")
        update_value_mass(kg, "value:relatedness", supports_inc=0.4, decay=0.01, note="journal_support")
        # decision report
        create_decision_report(
            kg,
            kind="journal",
            reason="short reflective journaling",
            confidence=0.6,
            result={"entry": str(p_txt), "summary": str(p_sum)},
            gains=["clarity", "self-connection"],
            tradeoffs=["time"],
            evidence_paths=[str(p_txt), str(p_sum)],
        )
        kg.save()
        # Reinforce desire state
        self.desire.reinforce({"clarity": 0.02, "relatedness": 0.01})
        self._stamp_action('journal')
        return "journal", {"entry": str(p_txt), "summary": str(p_sum), "confidence": 0.6}

    def _do_creative(self, payload: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        genre = payload.get('genre', 'story')
        theme = payload.get('theme', 'growth')
        cwc = CreativeWritingCortex()
        scenes = cwc.write_story(genre, theme, beats=4, words_per_scene=60)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path("data/writings")
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / f"auto_{ts}_{genre}_{theme}.md"
        lines = [f"# {genre.title()} — Theme: {theme}", "", "## Scenes"]
        for s in scenes:
            lines += [f"\n### {s.index}. {s.title}", s.content]
        path.write_text("\n".join(lines), encoding="utf-8")
        kg = KGManager()
        story_id = f"story_{ts}_auto"
        kg.add_node(story_id, properties={"type": "story", "genre": genre, "theme": theme, "path": str(path)})
        add_value_relation(kg, story_id, "value:creativity", "supports", confidence=0.65, evidence_paths=[str(path)], note="self-expression")
        update_value_mass(kg, "value:creativity", supports_inc=0.65, decay=0.01, note="creative_support")
        create_decision_report(
            kg,
            kind="creative",
            reason="small auto story",
            confidence=0.65,
            result={"story": str(path)},
            gains=["creativity"],
            tradeoffs=["time"],
            evidence_paths=[str(path)],
        )
        kg.save()
        self.desire.reinforce({"creativity": 0.02})
        self._stamp_action('creative')
        return "creative", {"story": str(path), "confidence": 0.65}

    def _do_math_verify(self, payload: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        stmt = payload.get('statement', '3*(2+4)=18')
        proof = MathCortex().verify(stmt)
        img = ProofRenderer().render(proof)
        kg = KGManager()
        kg.add_node("math_proof_auto", properties={"type": "artifact", "experience_visual": [img]})
        add_value_relation(kg, "math_proof_auto", "value:verifiability", "supports", confidence=0.7, evidence_paths=[img], note="explained proof")
        update_value_mass(kg, "value:verifiability", supports_inc=0.7, decay=0.01, note="math_support")
        create_decision_report(
            kg,
            kind="math_verify",
            reason="verify equality with explanation",
            confidence=0.7,
            result={"image": img, "valid": bool(proof.valid)},
            gains=["verifiability"],
            tradeoffs=["time"],
            evidence_paths=[img],
        )
        kg.save()
        self.desire.reinforce({"verifiability": 0.02})
        self._stamp_action('math_verify')
        return "math_verify", {"valid": proof.valid, "image": img, "confidence": 0.7}
