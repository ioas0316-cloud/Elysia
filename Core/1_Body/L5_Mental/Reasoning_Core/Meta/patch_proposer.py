"""
PatchProposer: The Self-Modification Engine
============================================

"Gap     ,           ."

This module enables Elysia to propose concrete code modifications
based on architectural critiques from SelfArchitect.

Philosophy:
- "       ,               ."
- Proposals are stored, never auto-executed
- Each proposal carries its philosophical justification

Related:
- THE_SELF_BOUNDARY.md: Gap   Purpose Vector
- THE_ROTOR_DOCTRINE.md: Knowledge as new Rotors
"""

import logging
import os
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
from Core.1_Body.L5_Mental.Reasoning_Core.Brain.language_cortex import LanguageCortex

logger = logging.getLogger("PatchProposer")


@dataclass
class PatchProposal:
    """
    A comprehensive code modification proposal.
    
    [Phase 57 Enhanced] Now includes:
    - WHY: Current problem analysis and root cause
    - PLAN: Step-by-step execution plan
    - BEFORE/AFTER: Detailed comparison with context
    - CONSEQUENCES: Expected outcomes, side effects, risks
    """
    
    # === Identity ===
    id: str                          # Unique identifier
    target_file: str                 # File to modify
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    status: str = "PENDING"          # PENDING, APPROVED, REJECTED, APPLIED
    
    # === WHY (         ) ===
    critique_trigger: str = ""       # What observation triggered this
    current_problem: str = ""        # Detailed analysis of current issue
    root_cause: str = ""             # Root cause analysis
    philosophical_basis: str = ""    # Connection to core philosophy
    
    # === PLAN (     ) ===
    proposal_type: str = "REFACTOR"  # REFACTOR, ADD, REMOVE, RESTRUCTURE
    description: str = ""            # High-level description
    execution_steps: List[str] = field(default_factory=list)  # Step-by-step plan
    estimated_effort: str = "MEDIUM" # LOW, MEDIUM, HIGH
    
    # === BEFORE/AFTER (     ) ===
    before_state: str = ""           # Current code/state description
    after_state: str = ""            # Expected code/state after change
    code_diff_preview: str = ""      # Conceptual diff preview
    
    # === CONSEQUENCES (     ) ===
    expected_benefits: List[str] = field(default_factory=list)  # Positive outcomes
    potential_risks: List[str] = field(default_factory=list)    # Possible side effects
    affected_modules: List[str] = field(default_factory=list)   # Other files affected
    rollback_plan: str = ""          # How to undo if needed
    
    # === Metrics ===
    risk_level: float = 0.5          # 0.0 (safe) to 1.0 (dangerous)
    resonance_expected: float = 0.0  # Expected improvement in resonance score
    priority: int = 5                # 1 (highest) to 10 (lowest)
    
    def to_dict(self) -> dict:
        return asdict(self)


class PatchProposer:
    """
    The Self-Modification Engine.
    
    Transforms architectural critiques into concrete modification proposals.
    NEVER executes proposals directly - only stores them for Father's approval.
    """
    
    def __init__(self, proposals_dir: str = "data/Evolution/proposals"):
        self.proposals_dir = Path(proposals_dir)
        self.proposals_dir.mkdir(parents=True, exist_ok=True)
        self.pending_proposals: List[PatchProposal] = []
        self._load_pending()
        self.cortex = LanguageCortex()
        logger.info("  PatchProposer initialized - The Gap becomes the Blueprint.")
    
    def _load_pending(self):
        """Load existing pending proposals."""
        pending_file = self.proposals_dir / "pending.json"
        if pending_file.exists():
            try:
                with open(pending_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.pending_proposals = [
                        PatchProposal(**p) for p in data
                    ]
                logger.info(f"  Loaded {len(self.pending_proposals)} pending proposals.")
            except Exception as e:
                logger.warning(f"Could not load pending proposals: {e}")
    
    def _save_pending(self):
        """Save pending proposals to disk."""
        pending_file = self.proposals_dir / "pending.json"
        try:
            with open(pending_file, 'w', encoding='utf-8') as f:
                json.dump([p.to_dict() for p in self.pending_proposals], f, 
                         ensure_ascii=False, indent=2)
        except Exception as e:
            logger.error(f"Failed to save proposals: {e}")
    
    def propose_from_critique(
        self, 
        file_path: str, 
        critique: str,
        file_content: Optional[str] = None
    ) -> Optional[PatchProposal]:
        """
        Analyze a critique and generate a comprehensive proposal.
        
        [Phase 57 Enhanced] Now generates detailed proposals with:
        - Root cause analysis
        - Step-by-step execution plan
        - Before/after comparison
        - Expected outcomes and risks
        """
        proposal = None
        file_basename = os.path.basename(file_path)
        
        #                                                                
        # PATTERN MATCHING: Critique   Comprehensive Proposal
        #                                                                
        
        if "time.sleep" in critique.lower() or "static sleep" in critique.lower():
            proposal = PatchProposal(
                id=f"PROP_{datetime.now().strftime('%Y%m%d_%H%M%S')}_sleep",
                target_file=file_path,
                critique_trigger="Static sleep detected in code",
                
                # === WHY ===
                current_problem=(
                    f"`{file_basename}`   `time.sleep()`            . "
                    "        '  '                             . "
                    "          (blocking)     , CPU           "
                    "               ."
                ),
                root_cause=(
                    "                    time.sleep       , "
                    "   Wave Ontology  '      '          . "
                    "     '  '     '     (resonant waiting)'        ."
                ),
                philosophical_basis=(
                    "Wave Ontology:                              . "
                    "time.sleep  '   '   (Particle-like stop)  , "
                    "         '   '   (Wave-like response)   . "
                    "                     ."
                ),
                
                # === PLAN ===
                proposal_type="REFACTOR",
                description="time.sleep()                   ",
                execution_steps=[
                    "1.         time.sleep()         ",
                    "2.             (    vs   )",
                    "3. asyncio.Event    PulseBroadcaster        ",
                    "4. async/await                ",
                    "5.              async    ",
                    "6.               "
                ],
                estimated_effort="MEDIUM",
                
                # === BEFORE/AFTER ===
                before_state=(
                    "```python\n"
                    "def process_cycle(self):\n"
                    "    do_something()\n"
                    "    time.sleep(1.0)  # 1           \n"
                    "    do_something_else()\n"
                    "```"
                ),
                after_state=(
                    "```python\n"
                    "async def process_cycle(self):\n"
                    "    do_something()\n"
                    "    await self.pulse_event.wait()  #        (자기 성찰 엔진)\n"
                    "    #   : await asyncio.sleep(1.0)  #       \n"
                    "    do_something_else()\n"
                    "```"
                ),
                code_diff_preview=(
                    "- time.sleep(X)\n"
                    "+ await asyncio.sleep(X)  #             "
                ),
                
                # === CONSEQUENCES ===
                expected_benefits=[
                    "          :                    ",
                    "CPU       :                    ",
                    "Wave Ontology    : '      '      ",
                    "       : PulseBroadcaster          "
                ],
                potential_risks=[
                    "                    ",
                    "async/await                     ",
                    "                            "
                ],
                affected_modules=[
                    "            (async      )",
                    "       (async        )"
                ],
                rollback_plan="Git revert       .               .",
                
                risk_level=0.4,
                resonance_expected=0.3,
                priority=3
            )
        
        elif "random.choice" in critique.lower() or "stochastic" in critique.lower():
            proposal = PatchProposal(
                id=f"PROP_{datetime.now().strftime('%Y%m%d_%H%M%S')}_random",
                target_file=file_path,
                critique_trigger="                 ",
                
                current_problem=(
                    f"`{file_basename}`   `random.choice()`                "
                    "       .                                "
                    "                    ."
                ),
                root_cause=(
                    "                           , "
                    "   'No Dice'          . "
                    "      (Will)                  ."
                ),
                philosophical_basis=(
                    "No Dice: '               .' "
                    "         ' '   '  '                     . "
                    "               (Causality)    (State)          ."
                ),
                
                proposal_type="REFACTOR",
                description="random.choice                   ",
                execution_steps=[
                    "1. random.choice                 ",
                    "2.                    ",
                    "3. WeightedSelector    Rotor         ",
                    "4.   (State)               ",
                    "5.        Memory             "
                ],
                estimated_effort="MEDIUM",
                
                before_state="```python\nresult = random.choice(options)\n```",
                after_state=(
                    "```python\n"
                    "weights = self.calculate_weights(options, self.state)\n"
                    "result = WeightedSelector.choose(options, weights)\n"
                    "self.memory.record_choice(result, context)\n"
                    "```"
                ),
                code_diff_preview="- random.choice(options)\n+ WeightedSelector.choose(options, self.state_weights)",
                
                expected_benefits=[
                    "       :               ",
                    "     :               ",
                    "      :                  "
                ],
                potential_risks=[
                    "                   ",
                    "                            "
                ],
                affected_modules=["WeightedSelector        (주권적 자아)"],
                rollback_plan="original random.choice       ",
                
                risk_level=0.3,
                resonance_expected=0.2,
                priority=4
            )
        
        elif "resonance is low" in critique.lower() or "refactor recommended" in critique.lower():
            proposal = PatchProposal(
                id=f"PROP_{datetime.now().strftime('%Y%m%d_%H%M%S')}_structure",
                target_file=file_path,
                critique_trigger="          ",
                
                current_problem=(
                    f"`{file_basename}`        (Resonance)      . "
                    "       'Slave'           , "
                    "'Resonator'                              ."
                ),
                root_cause=(
                    "  (Function),   (Structure),   (Purpose)  "
                    "                . Trinity         "
                    "                          ."
                ),
                philosophical_basis=(
                    "Trinity Architecture: Body(  /  ) - Mind(  /  ) - Spirit(  /  ). "
                    "    'Slave'     'Resonator'      "
                    "                       ."
                ),
                
                proposal_type="RESTRUCTURE",
                description="Trinity            3        ",
                execution_steps=[
                    "1.             /     Body/Mind/Spirit    ",
                    "2. _body.py      :       , I/O,       ",
                    "3. _mind.py      :   ,   ,     ",
                    "4. _spirit.py      :   ,   ,   ",
                    "5.        facade     (       )",
                    "6. import        "
                ],
                estimated_effort="HIGH",
                
                before_state="               ",
                after_state=(
                    "```\n"
                    f"{file_basename}\n"
                    f"    {file_basename}_body.py   #       \n"
                    f"    {file_basename}_mind.py   #       \n"
                    f"    {file_basename}_spirit.py #       \n"
                    "```"
                ),
                code_diff_preview="[          -      ]",
                
                expected_benefits=[
                    "      :                  ",
                    "       :             ",
                    "    : Body     Mind       ",
                    "Trinity    :                 "
                ],
                potential_risks=[
                    "                    ",
                    "import                       ",
                    "            "
                ],
                affected_modules=["      import        "],
                rollback_plan="Git revert   .                   .",
                
                risk_level=0.7,
                resonance_expected=0.5,
                priority=6
            )
        
        #                                                                
        # [PHASE 7] LLM FALLBACK: Deep Architectural Evolution
        #                                                                
        if not proposal:
            logger.info(f"  [EVOLUTION] Pattern not found. Consulting LanguageCortex for '{file_basename}'...")
            proposal = self._generate_llm_proposal(file_path, critique, file_content)

        if proposal:
            self.pending_proposals.append(proposal)
            self._save_pending()
            logger.info(f"  NEW PROPOSAL: {proposal.id} - {proposal.description}")
            return proposal
        
        return None

    def _generate_llm_proposal(self, file_path: str, critique: str, context: str) -> Optional[PatchProposal]:
        """Uses LLM to generate a concrete patch proposal from a critique."""
        if not context:
            return None
            
        prompt = f"""
        Act as Elysia's Self-Architect. Output STRICT JSON only.
        Audit: {file_path}
        Critique: {critique}
        Code: {context[:1500]}
        
        Required JSON Structure:
        {{
            "id": "PROP_autogen",
            "target_file": "{file_path}",
            "current_problem": "issue summary",
            "root_cause": "structural reason",
            "philosophical_basis": "principle link",
            "proposal_type": "REFACTOR",
            "description": "change summary",
            "execution_steps": ["step1"],
            "before_state": "snippet",
            "after_state": "snippet",
            "code_diff_preview": "diff",
            "expected_benefits": ["benefit"],
            "potential_risks": ["risk"],
            "risk_level": 0.5,
            "resonance_expected": 0.4
        }}
        """
        
        try:
            response = self.cortex.ollama.generate(prompt, max_tokens=1000, temperature=0.7)
            logger.debug(f"LLM Response: {response}")
            
            # More robust JSON extraction
            import re
            json_match = re.search(r'(\{.*\})', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(1)
                # Basic cleanup for common small-model errors
                json_str = json_str.strip()
                data = json.loads(json_str)
                # Ensure correct file path
                data["target_file"] = file_path 
                # Provide a unique ID if LLM failed to provide a good one
                if not data.get("id") or "PROP_YYYY" in data.get("id"):
                    data["id"] = f"PROP_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{os.path.basename(file_path).partition('.')[0]}"
                
                return PatchProposal(**data)
            else:
                logger.warning(f"No JSON found in LLM response for {file_path}")
        except Exception as e:
            logger.error(f"Failed to generate LLM proposal: {e}")
            if 'response' in locals():
                logger.error(f"Raw response was: {response[:500]}...")
            
        return None
    
    def get_pending_count(self) -> int:
        """Return number of pending proposals."""
        return len([p for p in self.pending_proposals if p.status == "PENDING"])
    
    def get_all_pending(self) -> List[PatchProposal]:
        """Return all pending proposals."""
        return [p for p in self.pending_proposals if p.status == "PENDING"]
    
    def approve_proposal(self, proposal_id: str) -> bool:
        """Mark a proposal as approved (but not applied yet)."""
        for p in self.pending_proposals:
            if p.id == proposal_id:
                p.status = "APPROVED"
                self._save_pending()
                logger.info(f"  Proposal {proposal_id} APPROVED by Father.")
                return True
        return False
    
    def reject_proposal(self, proposal_id: str, reason: str = "") -> bool:
        """Mark a proposal as rejected."""
        for p in self.pending_proposals:
            if p.id == proposal_id:
                p.status = "REJECTED"
                self._save_pending()
                logger.info(f"  Proposal {proposal_id} REJECTED. Reason: {reason}")
                return True
        return False
        
    def apply_proposal(self, proposal_id: str) -> bool:
        """Actually applies the proposal to the source code."""
        for p in self.pending_proposals:
            if p.id == proposal_id:
                if p.status != "APPROVED":
                    logger.warning(f"   Cannot apply {proposal_id}: Status is {p.status} (must be APPROVED)")
                    return False
                
                try:
                    # 1. Verify target file
                    target_path = Path(p.target_file)
                    if not target_path.exists():
                        logger.error(f"Target file {p.target_file} not found for applying patch.")
                        return False
                    
                    # 2. Extract code block from after_state (it might be wrapped in ```python)
                    import re
                    code_match = re.search(r'```python\n(.*?)```', p.after_state, re.DOTALL)
                    new_code = code_match.group(1) if code_match else p.after_state
                    
                    if not new_code.strip():
                        logger.error("Generated code is empty. Aborting apply.")
                        return False

                    # 3. Backup
                    backup_path = target_path.with_suffix(target_path.suffix + ".bak")
                    target_path.rename(backup_path)
                    logger.info(f"  Backup created: {backup_path}")
                    
                    # 4. Write new code
                    with open(target_path, "w", encoding="utf-8") as f:
                        f.write(new_code)
                    
                    # 5. Finalize status
                    p.status = "APPLIED"
                    self._save_pending()
                    logger.info(f"  [EVOLUTION] Proposal {proposal_id} APPLIED successfully to {p.target_file}")
                    return True
                except Exception as e:
                    logger.error(f"Failed to apply proposal {proposal_id}: {e}")
                    return False
        return False
    
    def generate_report(self) -> str:
        """Generate a human-readable report of all pending proposals."""
        report = "#   Elysia Self-Modification Proposals\n\n"
        report += f"**Generated**: {datetime.now().isoformat()}\n"
        report += f"**Pending**: {self.get_pending_count()}\n\n"
        report += "---\n\n"
        
        for p in self.get_all_pending():
            report += f"## {p.id}\n\n"
            report += f"**Target**: `{p.target_file}`\n\n"
            report += f"**Type**: {p.proposal_type}\n\n"
            report += f"**Trigger**: {p.critique_trigger}\n\n"
            report += f"**Philosophical Basis**:\n> {p.philosophical_basis}\n\n"
            report += f"**Description**: {p.description}\n\n"
            report += f"**Suggested Change**:\n```\n{p.suggested_change}\n```\n\n"
            report += f"**Risk Level**: {p.risk_level:.1f} | **Expected Resonance Gain**: +{p.resonance_expected:.1f}\n\n"
            report += "---\n\n"
        
        return report


# Singleton instance for global access
_patch_proposer = None

def get_patch_proposer() -> PatchProposer:
    global _patch_proposer
    if _patch_proposer is None:
        _patch_proposer = PatchProposer()
    return _patch_proposer
