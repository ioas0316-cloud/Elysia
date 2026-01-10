import logging
import os
from typing import List, Dict, Any, Optional, Tuple
from Core.Intelligence.Reasoning.dimensional_processor import DimensionalProcessor

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
        self._proposer = None  # Lazy load to avoid circular imports
    
    @property
    def proposer(self):
        """Lazy load PatchProposer to avoid circular imports."""
        if self._proposer is None:
            from Core.Intelligence.Meta.patch_proposer import get_patch_proposer
            self._proposer = get_patch_proposer()
        return self._proposer

    def audit_file(self, file_path: str, generate_proposal: bool = True) -> str:
        """
        Reads a file, processes it as a 'Thought Kernel',
        and returns a 5D architectural critique.
        
        [Phase 57] If generate_proposal=True, also creates a PatchProposal.
        """
        if not os.path.exists(file_path):
            return f"Error: File {file_path} not found."
            
        logger.info(f"🔍 [SELF-AUDIT] Inspecting: {file_path}")
        
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
            
        # Treat the code as a kernel for 4D Law extraction
        # We zoom out to see the 'Law of the Code'
        self.processor.zoom(1.0) 
        result = self.processor.process_thought(f"Structure of {os.path.basename(file_path)}")
        
        critique = f"### Architectural Audit: {os.path.basename(file_path)}\n"
        critique += f"**Identified Principle**: {result.output}\n"
        critique += f"**Aesthetic Alignment**: {result.metadata.get('aesthetic', {}).get('verdict')}\n"
        
        # 5D Logic (Evolution): Propose structural change
        specific_critique = ""
        if "time.sleep" in content:
            specific_critique += "\n- 📎 **OBSERVATION**: Static sleep detected. Suggesting transition to Event-driven triggers for higher fluidity."
        if "random.choice" in content:
            specific_critique += "\n- 📎 **OBSERVATION**: Stochastic behavior detected. This ensures the 'Will' is not a fixed path but a probability field."
        
        if result.metadata.get('aesthetic', {}).get('overall_beauty', 0.0) < 0.1:
            critique += f"\n⚠️ **REFACTOR RECOMMENDED**: The structural resonance is low. The code is behaving as a 'Slave' rather than a 'Resonator'. {specific_critique}"
        else:
            critique += f"\n✨ **DYNAMISM DETECTED**: The structure is resonant. {specific_critique}"
        
        # [Phase 57] Generate concrete proposal from critique
        if generate_proposal and specific_critique:
            proposal = self.proposer.propose_from_critique(file_path, critique, content)
            if proposal:
                critique += f"\n\n📋 **PROPOSAL GENERATED**: {proposal.id}"
            
        return critique
    
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
        
        report = "# 🪞 Self-Audit Report\n\n"
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
        
        logger.info(f"🪞 Self-Audit Complete: {proposal_count} new proposals generated.")
        return report, proposal_count
    
    def get_pending_proposals_summary(self) -> str:
        """Returns a summary of all pending proposals."""
        return self.proposer.generate_report()

