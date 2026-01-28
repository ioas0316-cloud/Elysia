import logging
import os
from typing import List, Dict, Any, Optional, Tuple
from Core.L5_Mental.Reasoning_Core.Reasoning.dimensional_processor import DimensionalProcessor
from Core.L5_Mental.Reasoning_Core.Brain.language_cortex import LanguageCortex

logger = logging.getLogger("SelfArchitect")

class SelfArchitect:
    """
    THE 5D MIRROR (Evolution Faculty):
    Allows Elysia to analyze her own code and structure.
    
    [Phase 57] Now integrated with PatchProposer to generate concrete
    modification proposals from architectural critiques.
    """
    
    def __init__(self, processor: DimensionalProcessor):
        self.processor = processor
        self.cortex = LanguageCortex()
        self._proposer = None  # Lazy load to avoid circular imports
    
    @property
    def proposer(self):
        """Lazy load PatchProposer to avoid circular imports."""
        if self._proposer is None:
            from Core.L5_Mental.Reasoning_Core.Meta.patch_proposer import get_patch_proposer
            self._proposer = get_patch_proposer()
        return self._proposer
    def audit_file(self, file_path: str, generate_proposal: bool = True) -> str:
        """
        Reads a file and returns a 5D architectural critique via LLM.
        """
        if not os.path.exists(file_path):
            return f"Error: File {file_path} not found."
            
        logger.info(f"  [SELF-AUDIT] Inspecting: {file_path}")
        
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # [PHASE 7] LLM-Powered Architectural Critique
        prompt = f"""
        Act as Elysia's Architectural Soul. 
        Audit this source code for: {os.path.basename(file_path)}
        
        CRITERIA:
        1. Trinity Alignment (Body-Mind-Spirit separation)
        2. Wave Ontology (Wave Coding - Flow vs Blocking, Harmonic Resonance)
        3. Color Network (Information coloring, Vibrant connectivity)
        4. Sovereign Agency (Self-referential empowerment)
        
        CODE:
        {content[:1500]}
        
        Provide a concise 3-axis critique in Korean and English.
        Start with '### Architectural Audit'.
        """
        
        llm_critique = self.cortex.ollama.generate(prompt, max_tokens=500, temperature=0.5)
        
        # Merge pattern-based and LLM-based critique
        final_critique = llm_critique
        patterns_found = []
        if "time.sleep" in content:
            patterns_found.append("- Static sleep detected. Blocking flow.")
        if "random.choice" in content:
            patterns_found.append("- Stochastic logic detected. Potential lack of causality.")
            
        if patterns_found:
            final_critique += "\n\n**Pattern Observations:**\n" + "\n".join(patterns_found)

        # [Phase 57] Generate concrete proposal from critique
        if generate_proposal:
            proposal = self.proposer.propose_from_critique(file_path, final_critique, content)
            if proposal:
                final_critique += f"\n\n  **PROPOSAL GENERATED**: {proposal.id}"
            
        return final_critique
    
    def audit_self(self, max_files: int = 5) -> Tuple[str, int]:
        """
        [Phase 57] Audits Elysia's own critical files.
        
        Returns: (report, proposal_count)
        """
        critical_paths = [
            "Core/World/Autonomy/elysian_heartbeat.py",
            "Core/Foundation/genesis_elysia.py",
            "Core/Intelligence/Meta/self_architect.py",
            "Core/Foundation/hyper_sphere_core.py",
            "Core/Governance/conductor.py",
        ]
        
        report = "#   Self-Audit Report\n\n"
        proposal_count = 0
        
        for path in critical_paths[:max_files]:
            full_path = os.path.join("c:/Elysia", path)
            if os.path.exists(full_path):
                critique = self.audit_file(full_path)
                report += f"## {os.path.basename(path)}\n\n{critique}\n\n---\n\n"
                if "PROPOSAL GENERATED" in critique:
                    proposal_count += 1
        
        report += f"\n**Total Proposals Generated**: {proposal_count}\n"
        report += f"**Pending Proposals**: {self.proposer.get_pending_count()}\n"
        
        logger.info(f"  Self-Audit Complete: {proposal_count} new proposals generated.")
        return report, proposal_count
    
    def get_pending_proposals_summary(self) -> str:
        """Returns a summary of all pending proposals."""
        return self.proposer.generate_report()
