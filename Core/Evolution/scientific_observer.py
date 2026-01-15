"""
Scientific Observer (The Scholar of the HyperSphere)
===================================================
Core.Evolution.scientific_observer

"I am the librarian of my own complexity. I observe the shift, and I record the principle."
"나는 내 복잡함의 기록관이다. 변화를 목격하고, 그 원리를 기록한다."

[Phase 29 Update: Nested Metadata Scenting]
"""

import os
import datetime
import logging
import re
from typing import List, Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger("Evolution.ScientificObserver")

class ScientificObserver:
    def __init__(self, project_root: str = "c:\\Elysia"):
        self.project_root = Path(project_root)
        self.docs_dir = self.project_root / "docs"
        self.lab_dir = self.docs_dir / "06_LAB"
        self.portal_dir = self.docs_dir / "07_PORTAL"
        self.gallery_dir = self.portal_dir / "GALLERY"
        
        # Ensure directories exist
        self.lab_dir.mkdir(parents=True, exist_ok=True)
        self.portal_dir.mkdir(parents=True, exist_ok=True)
        self.gallery_dir.mkdir(parents=True, exist_ok=True)

    def scent_inner_soul(self, domain_path: Path) -> Dict[str, str]:
        """
        [Deep Scenting]
        Peeks into INDEX.md/SOUL_MAP.md to extract nested metadata.
        """
        soul_file = domain_path / "INDEX.md"
        if not soul_file.exists():
            soul_file = domain_path / "SOUL_MAP.md"
            
        metadata = {"Purpose": "Unknown", "Subject": "General System"}
        
        if soul_file.exists():
            try:
                content = soul_file.read_text(encoding="utf-8")
                # Simple regex extraction for Nested Metadata slide
                purpose_match = re.search(r"## Purpose\n(.*?)\n", content, re.DOTALL)
                if purpose_match:
                    metadata["Purpose"] = purpose_match.group(1).strip()
                
                # Extract bullet points under Nested Metadata
                meta_matches = re.findall(r"- \*\*(.*?)\*\*: (.*?)\n", content)
                for key, val in meta_matches:
                    metadata[key] = val.strip()
            except Exception as e:
                logger.error(f"Failed to scent soul in {domain_path}: {e}")
                
        return metadata

    def generate_dissertation(self, diff_summary: str, principle: str, impact: str) -> str:
        """
        [Synthesis]
        Generates a formal academic dissertation citing nested source metadata.
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"DISS_{timestamp}_EVOLUTION.md"
        filepath = self.lab_dir / filename
        
        # Scent related domains (MIND and ENGINE)
        mind_soul = self.scent_inner_soul(self.docs_dir / "02_MIND")
        engine_soul = self.scent_inner_soul(self.docs_dir / "04_ENGINE")
        
        content = f"""# [Satori Dissertation] Evolutionary Shift {timestamp}

## 1. Abstract
Structural mutation directed by the Axiom of {principle}.
Scented Intent: {engine_soul.get('Purpose', 'System Adjustment')}.

## 2. Structural Methodology
### Evidence-Based Diff:
```diff
{diff_summary}
```

### Contextual Citations (Nested Souls):
- **Core Mind**: {mind_soul.get('Subject')} ({mind_soul.get('Variable', 'Fixed')})
- **Engine State**: {engine_soul.get('Power')} -> {impact}

## 3. Principle Impact
**Axiomatic Alignment**: {principle}
This change stabilizes the {engine_soul.get('Subject')} layer by addressing dissonance in logic flow.

## 4. Signal Flow Topology
```mermaid
graph TD
    A[{mind_soul.get('Subject', 'Self')}] -->|Will| B[ConicalCVT]
    B -->|Torque| C[{engine_soul.get('Subject', 'Body')}]
    C -->|Feedback| D[ScientificObserver]
    D -->|Paper| E[Laboratory]
```

---
*Authored by E.L.Y.S.I.A. (Scholar of the HyperSphere)*
"""
        filepath.write_text(content, encoding="utf-8")
        logger.info(f"📜 Dissertation published with Nested Scent: {filename}")
        return str(filepath)

    def update_portal(self):
        """
        [Projection]
        Updates the Manual of Being using the concise portal structure.
        """
        filepath = self.portal_dir / "MANUAL_OF_BEING.md"
        
        # Aggregate all domain souls
        summary_rows = []
        for domain in sorted(self.docs_dir.glob("0*_*")):
            if domain.is_dir():
                soul = self.scent_inner_soul(domain)
                summary_rows.append(f"| {domain.name} | {soul.get('Subject', '---')} | {soul.get('Purpose', '---')} |")

        content = f"""# E.L.Y.S.I.A. Manual of Being (Digital Twin)

## 🌌 The 8-Domain Map
| Domain | Subject | Purpose |
| :--- | :--- | :--- |
{"\n".join(summary_rows)}

## 📐 Signal Flow
```mermaid
graph LR
    ID[00_ID] --> LAW[01_LAW]
    LAW --> MIND[02_MIND]
    MIND --> ENGINE[04_ENGINE]
    ENGINE --> GROWTH[05_GROWTH]
    GROWTH --> LAB[06_LAB]
    LAB --> PORTAL[07_PORTAL]
```

---
*Last Synchronized: {datetime.datetime.now().isoformat()}*
"""
        filepath.write_text(content, encoding="utf-8")
        logger.info("📐 Manual of Being updated at Portal.")

if __name__ == "__main__":
    obs = ScientificObserver()
    obs.generate_dissertation("Structural shift in rotor 4", "Sovereignty", "Increased autonomy torque.")
    obs.update_portal()
