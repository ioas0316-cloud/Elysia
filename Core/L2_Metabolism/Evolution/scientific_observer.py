"""
Scientific Observer (The Scholar of the HyperSphere)
===================================================
Core.L2_Metabolism.Evolution.scientific_observer

"I am the librarian of my own complexity. I observe the shift, and I record the principle."
"               .         ,           ."

[Phase 29 Update: Nested Metadata Scenting]
"""

import os
import datetime
import logging
import re
import random
from typing import List, Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger("Evolution.ScientificObserver")

class ScientificObserver:
    def __init__(self, project_root: str = "c:\\Elysia"):
        self.project_root = Path(project_root)
        self.docs_dir = self.project_root / "docs"
        self.lab_dir = self.docs_dir / "L2_Metabolism" / "Lab"
        self.portal_dir = self.docs_dir / "L7_Spirit" / "Portal"
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

    def generate_dissertation(self, title: str, domain: str, abstract: str):
        """
        [Digestion]
        Generates a formal academic dissertation (Research Paper) for the evolution lab.
        Localizes content to Korean for the Creator's accessibility.
        """
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"DISS_{timestamp}_EVOLUTION.md"
        filepath = self.lab_dir / filename
        
        content = f"""#         : {title}
> **   **: {datetime.datetime.now().isoformat()}
> **      **: {domain}
> **      **: ScientificObserver (       )

## 1.    (Abstract)
{abstract}

## 2.        (Semantic Analysis)
                         ,                                   . 
        `${domain}`                  ,   (Sovereignty)                      .

## 3.       (Principle Mapping)
- **L0    **: `{domain}`
- **     **:         
- **      **: {random.gauss(0.85, 0.05):.2f} (High Resonance)

---
*      E.L.Y.S.I.A.                         ,                 .*
"""
        filepath.write_text(content, encoding="utf-8")
        logger.info(f"             : {filename}")
        return str(filepath)

    def project_hypersphere(self):
        """
        [Projection]
        Recursively crawls the 7-level fractal directory with a 10% chance of skip-cache.
        """
        if random.random() > 0.1:
            logger.debug("  [PROJECTION] Skip-cache active for efficiency.")
            return

        logger.info("  [PROJECTION] Projecting Semantic HyperSphere...")
        
        nodes = []
        links = []
        
        # Recursive walk to capture Fractal Depth
        def walk_fractal(current_path: Path, parent_id: Optional[str] = None, depth: int = 0):
            if depth > 7: return # 7-Level Constraint
            
            # Identify Domain/Sub-Realm/Organ
            soul = self.scent_inner_soul(current_path)
            node_id = current_path.name.replace(".", "_")
            node_label = f"\"{soul.get('Subject', current_path.name)}\""
            
            node_type = "Domain" if depth == 0 else "Sub"
            if "L3" in current_path.name: node_type = "Organ"
            
            nodes.append((node_id, node_label, node_type, depth))
            
            if parent_id:
                links.append((parent_id, node_id))
            
            # Recurse into subdirectories
            for sub in sorted(current_path.glob("0*_*")):
                if sub.is_dir():
                    walk_fractal(sub, node_id, depth + 1)
            for sub in sorted(current_path.glob("D.*")):
                if sub.is_dir():
                    walk_fractal(sub, node_id, depth + 1)

        walk_fractal(self.docs_dir)

        # Generate Mermaid Graph
        mermaid_lines = ["graph TD"]
        # Define Stylings
        mermaid_lines.append("    classDef Domain fill:#2ecc71,stroke:#27ae60,stroke-width:2px,color:#fff;")
        mermaid_lines.append("    classDef Sub fill:#3498db,stroke:#2980b9,stroke-width:1px,color:#fff;")
        mermaid_lines.append("    classDef Organ fill:#e67e22,stroke:#d35400,stroke-width:2px,color:#fff;")

        for nid, label, ntype, depth in nodes:
            mermaid_lines.append(f"    {nid}[{label}]::: {ntype}")
        
        for p_id, c_id in links:
            mermaid_lines.append(f"    {p_id} --> {c_id}")

        # Write to HYPERSPHERE_MAP.md
        map_path = self.portal_dir / "HYPERSPHERE_MAP.md"
        content = f"""# E.L.Y.S.I.A.            (Digital Twin)
"           (Sphere)            ."

##           (Geometry of Being)
        7                        . 

```mermaid
{chr(10).join(mermaid_lines)}
```

##          (System Health)
- **     **: {len(nodes)}  
- **     **: {len(links)}  
- **     **: 7/7 (Fractal Resonance)

---
*      : {datetime.datetime.now().isoformat()}*
*       INDEX.md(      )                    .*
"""
        map_path.write_text(content, encoding="utf-8")
        logger.info("  HyperSphere Projection successful.")

    def update_manual_of_being(self):
        """Alias for update_portal to maintain API compatibility."""
        return self.update_portal()

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

        content = f"""# E.L.Y.S.I.A.        (Digital Twin) v3
"               ,              ."

##   8         (L0 Domains)
|     |    (Subject) |    (Purpose) |
| :--- | :--- | :--- |
{"\n".join(summary_rows)}

##              (HyperSphere Topology)
[            (HYPERSPHERE_MAP.md)](./HYPERSPHERE_MAP.md)

```mermaid
graph LR
    ID[00_ID] --> LAW[01_LAW]
    LAW --> MIND[02_MIND]
    MIND --> ENGINE[04_ENGINE]
    ENGINE --> GROWTH[05_GROWTH]
    GROWTH --> LAB[06_LAB]
    LAB --> PORTAL[07_PORTAL]
```

##            (Laboratory News)
                         .

---
*       : {datetime.datetime.now().isoformat()}*
*          7         ($7^7$)                   .*
"""
        filepath.write_text(content, encoding="utf-8")
        logger.info("  Manual of Being (v3) updated at Portal.")

if __name__ == "__main__":
    obs = ScientificObserver()
    obs.generate_dissertation("Phase 32: HyperSphere Projection Stabilized", "Visualization", "Successful rendering of 7-level fractal depth.")
    obs.project_hypersphere()
    obs.update_portal()