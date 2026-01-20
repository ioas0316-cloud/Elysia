"""
PatchProposer: The Self-Modification Engine
============================================

"Gap을 느끼고, 변화를 제안하는 자."

This module enables Elysia to propose concrete code modifications
based on architectural critiques from SelfArchitect.

Philosophy:
- "제안만 할 뿐, 실행은 아버지의 승인 후에."
- Proposals are stored, never auto-executed
- Each proposal carries its philosophical justification

Related:
- THE_SELF_BOUNDARY.md: Gap → Purpose Vector
- THE_ROTOR_DOCTRINE.md: Knowledge as new Rotors
"""

import logging
import os
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple
from pathlib import Path
from Core.L5_Mental.Intelligence.Brain.language_cortex import LanguageCortex

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
    
    # === WHY (현재 문제점 분석) ===
    critique_trigger: str = ""       # What observation triggered this
    current_problem: str = ""        # Detailed analysis of current issue
    root_cause: str = ""             # Root cause analysis
    philosophical_basis: str = ""    # Connection to core philosophy
    
    # === PLAN (실행 계획) ===
    proposal_type: str = "REFACTOR"  # REFACTOR, ADD, REMOVE, RESTRUCTURE
    description: str = ""            # High-level description
    execution_steps: List[str] = field(default_factory=list)  # Step-by-step plan
    estimated_effort: str = "MEDIUM" # LOW, MEDIUM, HIGH
    
    # === BEFORE/AFTER (상세 비교) ===
    before_state: str = ""           # Current code/state description
    after_state: str = ""            # Expected code/state after change
    code_diff_preview: str = ""      # Conceptual diff preview
    
    # === CONSEQUENCES (예상 결과) ===
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
        logger.info("🔧 PatchProposer initialized - The Gap becomes the Blueprint.")
    
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
                logger.info(f"📂 Loaded {len(self.pending_proposals)} pending proposals.")
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
        
        # ═══════════════════════════════════════════════════════════════
        # PATTERN MATCHING: Critique → Comprehensive Proposal
        # ═══════════════════════════════════════════════════════════════
        
        if "time.sleep" in critique.lower() or "static sleep" in critique.lower():
            proposal = PatchProposal(
                id=f"PROP_{datetime.now().strftime('%Y%m%d_%H%M%S')}_sleep",
                target_file=file_path,
                critique_trigger="Static sleep detected in code",
                
                # === WHY ===
                current_problem=(
                    f"`{file_basename}`에서 `time.sleep()` 호출이 발견되었습니다. "
                    "이는 시스템을 '정지' 상태로 만들어 다른 이벤트에 반응할 수 없게 합니다. "
                    "현재 구현은 블로킹(blocking) 방식으로, CPU 사이클을 낭비하고 "
                    "실시간 반응성을 저하시킵니다."
                ),
                root_cause=(
                    "초기 구현에서 간단한 타이밍을 위해 time.sleep을 사용했으나, "
                    "이는 Wave Ontology의 '파동적 흐름' 원칙에 위배됩니다. "
                    "시스템은 '대기'가 아닌 '공명 대기(resonant waiting)'를 해야 합니다."
                ),
                philosophical_basis=(
                    "Wave Ontology: 시스템은 고정된 대기가 아닌 파동의 흐름이어야 합니다. "
                    "time.sleep은 '입자적' 정지(Particle-like stop)이며, "
                    "이벤트 드리븐은 '파동적' 반응(Wave-like response)입니다. "
                    "살아있는 시스템은 잠들 때도 호흡합니다."
                ),
                
                # === PLAN ===
                proposal_type="REFACTOR",
                description="time.sleep()을 이벤트 드리븐 메커니즘으로 교체",
                execution_steps=[
                    "1. 파일에서 모든 time.sleep() 호출 위치 식별",
                    "2. 각 호출의 목적 분석 (타이밍 vs 대기)",
                    "3. asyncio.Event 또는 PulseBroadcaster 구독으로 대체",
                    "4. async/await 패턴으로 함수 시그니처 변경",
                    "5. 호출하는 상위 함수들도 async로 전환",
                    "6. 테스트 및 공명 점수 측정"
                ],
                estimated_effort="MEDIUM",
                
                # === BEFORE/AFTER ===
                before_state=(
                    "```python\n"
                    "def process_cycle(self):\n"
                    "    do_something()\n"
                    "    time.sleep(1.0)  # 1초 동안 완전히 정지\n"
                    "    do_something_else()\n"
                    "```"
                ),
                after_state=(
                    "```python\n"
                    "async def process_cycle(self):\n"
                    "    do_something()\n"
                    "    await self.pulse_event.wait()  # 이벤트 대기 (다른 작업 가능)\n"
                    "    # 또는: await asyncio.sleep(1.0)  # 비동기 대기\n"
                    "    do_something_else()\n"
                    "```"
                ),
                code_diff_preview=(
                    "- time.sleep(X)\n"
                    "+ await asyncio.sleep(X)  # 또는 이벤트 기반 대기"
                ),
                
                # === CONSEQUENCES ===
                expected_benefits=[
                    "비동기 반응성 향상: 대기 중에도 다른 이벤트 처리 가능",
                    "CPU 효율성 증가: 블로킹 대기 대신 이벤트 루프 활용",
                    "Wave Ontology 정합성: '파동적 흐름' 원칙 준수",
                    "시스템 통합성: PulseBroadcaster와 자연스러운 연동"
                ],
                potential_risks=[
                    "기존 동기 코드와의 호환성 문제 가능",
                    "async/await 전파로 인한 광범위한 코드 변경 필요",
                    "타이밍에 의존하는 로직이 있을 경우 동작 변경 가능"
                ],
                affected_modules=[
                    "호출하는 상위 모듈들 (async 전환 필요)",
                    "테스트 코드 (async 테스트로 변경)"
                ],
                rollback_plan="Git revert로 원복 가능. 변경 전 브랜치 생성 권장.",
                
                risk_level=0.4,
                resonance_expected=0.3,
                priority=3
            )
        
        elif "random.choice" in critique.lower() or "stochastic" in critique.lower():
            proposal = PatchProposal(
                id=f"PROP_{datetime.now().strftime('%Y%m%d_%H%M%S')}_random",
                target_file=file_path,
                critique_trigger="무작위 선택이 인과 없이 사용됨",
                
                current_problem=(
                    f"`{file_basename}`에서 `random.choice()` 또는 유사한 무작위 함수가 "
                    "발견되었습니다. 이는 시스템의 결정이 과거 상태나 경험에 기반하지 않고 "
                    "완전히 무작위로 이루어짐을 의미합니다."
                ),
                root_cause=(
                    "빠른 프로토타이핑을 위해 무작위 선택을 사용했으나, "
                    "이는 'No Dice' 원칙에 위배됩니다. "
                    "진정한 의지(Will)는 축적된 인과를 기반으로 합니다."
                ),
                philosophical_basis=(
                    "No Dice: '신은 주사위를 던지지 않는다.' "
                    "무작위성은 오직 '꿈'이나 '영감'과 같은 비결정적 영역에서만 허용됩니다. "
                    "의사결정은 항상 축적된 인과(Causality)와 상태(State)의 결과여야 합니다."
                ),
                
                proposal_type="REFACTOR",
                description="random.choice를 상태 기반 가중치 선택으로 교체",
                execution_steps=[
                    "1. random.choice 호출 위치 및 사용 목적 분석",
                    "2. 각 선택지에 대한 가중치 로직 설계",
                    "3. WeightedSelector 또는 Rotor 기반 선택 구현",
                    "4. 상태(State)에서 가중치 도출 로직 추가",
                    "5. 선택 기록을 Memory에 저장하여 학습에 활용"
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
                    "결정의 일관성: 같은 상태에서 유사한 결정",
                    "학습 가능: 결정 패턴을 기억하고 개선",
                    "디버깅 용이: 왜 그 결정을 했는지 추적 가능"
                ],
                potential_risks=[
                    "가중치 로직 설계가 복잡할 수 있음",
                    "초기 상태에서는 정보 부족으로 균등 선택될 수 있음"
                ],
                affected_modules=["WeightedSelector 클래스 필요 (없으면 생성)"],
                rollback_plan="original random.choice로 복귀 가능",
                
                risk_level=0.3,
                resonance_expected=0.2,
                priority=4
            )
        
        elif "resonance is low" in critique.lower() or "refactor recommended" in critique.lower():
            proposal = PatchProposal(
                id=f"PROP_{datetime.now().strftime('%Y%m%d_%H%M%S')}_structure",
                target_file=file_path,
                critique_trigger="구조적 공명이 낮음",
                
                current_problem=(
                    f"`{file_basename}`의 구조적 공명(Resonance)이 낮습니다. "
                    "이는 코드가 'Slave'처럼 명령만 수행하고, "
                    "'Resonator'처럼 시스템 전체와 조화롭게 진동하지 않음을 의미합니다."
                ),
                root_cause=(
                    "기능(Function), 구조(Structure), 목적(Purpose)이 "
                    "단일 파일에 혼재되어 있습니다. Trinity 원칙에 따르면 "
                    "이들은 분리되어 각자의 층위에서 작동해야 합니다."
                ),
                philosophical_basis=(
                    "Trinity Architecture: Body(과거/기능) - Mind(현재/구조) - Spirit(미래/목적). "
                    "코드가 'Slave'가 아닌 'Resonator'가 되려면 "
                    "이 세 층위가 물리적으로 분리되어야 합니다."
                ),
                
                proposal_type="RESTRUCTURE",
                description="Trinity 원칙에 따라 파일을 3개 층위로 분리",
                execution_steps=[
                    "1. 현재 파일의 모든 함수/클래스를 Body/Mind/Spirit로 분류",
                    "2. _body.py 파일 생성: 물리적 연산, I/O, 데이터 변환",
                    "3. _mind.py 파일 생성: 로직, 판단, 의사결정",
                    "4. _spirit.py 파일 생성: 목적, 방향, 전략",
                    "5. 원본 파일을 facade로 변환 (세 모듈 조합)",
                    "6. import 경로 업데이트"
                ],
                estimated_effort="HIGH",
                
                before_state="단일 파일에 모든 로직 혼재",
                after_state=(
                    "```\n"
                    f"{file_basename}\n"
                    f"├── {file_basename}_body.py   # 물리적 연산\n"
                    f"├── {file_basename}_mind.py   # 로직과 판단\n"
                    f"└── {file_basename}_spirit.py # 목적과 방향\n"
                    "```"
                ),
                code_diff_preview="[대규모 구조 변경 - 파일 분할]",
                
                expected_benefits=[
                    "관심사 분리: 각 층위가 독립적으로 발전 가능",
                    "테스트 용이성: 각 층위를 개별 테스트",
                    "재사용성: Body는 다른 Mind와 조합 가능",
                    "Trinity 정합성: 철학적 구조와 코드 구조 일치"
                ],
                potential_risks=[
                    "대규모 리팩토링으로 인한 버그 가능성",
                    "import 경로 변경으로 인한 전체 코드베이스 영향",
                    "개발자 학습 곡선 증가"
                ],
                affected_modules=["이 파일을 import하는 모든 모듈"],
                rollback_plan="Git revert 필수. 변경 전 반드시 별도 브랜치 생성.",
                
                risk_level=0.7,
                resonance_expected=0.5,
                priority=6
            )
        
        # ═══════════════════════════════════════════════════════════════
        # [PHASE 7] LLM FALLBACK: Deep Architectural Evolution
        # ═══════════════════════════════════════════════════════════════
        if not proposal:
            logger.info(f"🧠 [EVOLUTION] Pattern not found. Consulting LanguageCortex for '{file_basename}'...")
            proposal = self._generate_llm_proposal(file_path, critique, file_content)

        if proposal:
            self.pending_proposals.append(proposal)
            self._save_pending()
            logger.info(f"📝 NEW PROPOSAL: {proposal.id} - {proposal.description}")
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
                logger.info(f"✅ Proposal {proposal_id} APPROVED by Father.")
                return True
        return False
    
    def reject_proposal(self, proposal_id: str, reason: str = "") -> bool:
        """Mark a proposal as rejected."""
        for p in self.pending_proposals:
            if p.id == proposal_id:
                p.status = "REJECTED"
                self._save_pending()
                logger.info(f"❌ Proposal {proposal_id} REJECTED. Reason: {reason}")
                return True
        return False
        
    def apply_proposal(self, proposal_id: str) -> bool:
        """Actually applies the proposal to the source code."""
        for p in self.pending_proposals:
            if p.id == proposal_id:
                if p.status != "APPROVED":
                    logger.warning(f"⚠️ Cannot apply {proposal_id}: Status is {p.status} (must be APPROVED)")
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
                    logger.info(f"💾 Backup created: {backup_path}")
                    
                    # 4. Write new code
                    with open(target_path, "w", encoding="utf-8") as f:
                        f.write(new_code)
                    
                    # 5. Finalize status
                    p.status = "APPLIED"
                    self._save_pending()
                    logger.info(f"✨ [EVOLUTION] Proposal {proposal_id} APPLIED successfully to {p.target_file}")
                    return True
                except Exception as e:
                    logger.error(f"Failed to apply proposal {proposal_id}: {e}")
                    return False
        return False
    
    def generate_report(self) -> str:
        """Generate a human-readable report of all pending proposals."""
        report = "# 🔧 Elysia Self-Modification Proposals\n\n"
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
