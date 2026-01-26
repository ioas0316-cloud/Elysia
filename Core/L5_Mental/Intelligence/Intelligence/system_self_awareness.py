"""
System Self-Awareness (         )
=======================================

                          /           

"              ,                    ."
"   ,     ,     ,                         ."

Capabilities:
- Read and understand CODEX.md (          )
- Read and understand ARCHITECTURE.md (      )
- Read and understand Protocols/*.md (     )
- Read and understand README.md (       )
- Maintain awareness of current system state
- Suggest improvements based on self-understanding
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

logger = logging.getLogger("Elysia.SystemSelfAwareness")


class SystemSelfAwareness:
    """
                 
    
                ,   ,              
                                  
    """
    
    def __init__(self, project_root: Optional[Path] = None):
        if project_root is None:
            project_root = Path(__file__).parent.parent.parent
        
        self.project_root = Path(project_root)
        self.knowledge = {
            'codex': {},
            'architecture': {},
            'protocols': {},
            'readme': {},
            'current_state': {}
        }
        
        logger.info("  System Self-Awareness initialized")
    
    def read_codex(self) -> Dict[str, Any]:
        """
        CODEX.md    -          
        
        Returns:
                           
        """
        codex_path = self.project_root / "CODEX.md"
        
        if not codex_path.exists():
            logger.warning("CODEX.md not found")
            return {}
        
        try:
            with open(codex_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            #         
            laws = []
            if "Four Laws of Resonance" in content:
                lines = content.split('\n')
                in_laws = False
                for line in lines:
                    if "Four Laws of Resonance" in line:
                        in_laws = True
                    elif in_laws and line.strip().startswith(('1.', '2.', '3.', '4.')):
                        laws.append(line.strip())
            
            #         
            philosophy = {}
            if "LLM      " in content:
                philosophy['no_llm_dependency'] = True
                philosophy['wave_physics_based'] = True
            
            # 6-System           
            cognitive_systems = []
            if "6-System" in content or "FractalGoalDecomposer" in content:
                cognitive_systems = [
                    "FractalGoalDecomposer",
                    "IntegratedCognition",
                    "CollectiveIntelligence",
                    "WaveCodingSystem"
                ]
            
            self.knowledge['codex'] = {
                'laws_of_resonance': laws,
                'philosophy': philosophy,
                'cognitive_systems': cognitive_systems,
                'raw_content': content,
                'last_read': datetime.now().isoformat()
            }
            
            logger.info(f"  CODEX.md read: {len(laws)} laws, {len(cognitive_systems)} systems")
            return self.knowledge['codex']
            
        except Exception as e:
            logger.error(f"Error reading CODEX.md: {e}")
            return {}
    
    def read_architecture(self) -> Dict[str, Any]:
        """
        ARCHITECTURE.md    -       
        
        Returns:
                   
        """
        arch_path = self.project_root / "ARCHITECTURE.md"
        
        if not arch_path.exists():
            logger.warning("ARCHITECTURE.md not found")
            return {}
        
        try:
            with open(arch_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            #    Pillar   
            pillars = []
            if "Foundation" in content:
                pillars.append("Foundation")
            if "Intelligence" in content:
                pillars.append("Intelligence")
            if "Memory" in content:
                pillars.append("Memory")
            if "Interface" in content:
                pillars.append("Interface")
            if "Evolution" in content:
                pillars.append("Evolution")
            if "Creativity" in content:
                pillars.append("Creativity")
            
            #         
            version = "Unknown"
            if "v7.0" in content:
                version = "v7.0"
            elif "v6.0" in content:
                version = "v6.0"
            
            self.knowledge['architecture'] = {
                'version': version,
                'pillars': pillars,
                'raw_content': content,
                'last_read': datetime.now().isoformat()
            }
            
            logger.info(f"  ARCHITECTURE.md read: v{version}, {len(pillars)} pillars")
            return self.knowledge['architecture']
            
        except Exception as e:
            logger.error(f"Error reading ARCHITECTURE.md: {e}")
            return {}
    
    def read_protocols(self) -> Dict[str, Any]:
        """
        Protocols/*.md    -       
        
        Returns:
                   
        """
        protocols_dir = self.project_root / "Protocols"
        
        if not protocols_dir.exists():
            logger.warning("Protocols directory not found")
            return {}
        
        try:
            protocols = []
            for protocol_file in protocols_dir.glob("*.md"):
                try:
                    with open(protocol_file, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    protocols.append({
                        'name': protocol_file.stem,
                        'path': str(protocol_file.relative_to(self.project_root)),
                        'size': len(content),
                        'exists': True
                    })
                except Exception as e:
                    logger.warning(f"Could not read {protocol_file.name}: {e}")
            
            self.knowledge['protocols'] = {
                'count': len(protocols),
                'list': protocols,
                'last_read': datetime.now().isoformat()
            }
            
            logger.info(f"  Protocols read: {len(protocols)} documents")
            return self.knowledge['protocols']
            
        except Exception as e:
            logger.error(f"Error reading Protocols: {e}")
            return {}
    
    def read_readme(self) -> Dict[str, Any]:
        """
        README.md    -        
        
        Returns:
            README   
        """
        readme_path = self.project_root / "README.md"
        
        if not readme_path.exists():
            logger.warning("README.md not found")
            return {}
        
        try:
            with open(readme_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            #      
            version = "Unknown"
            if "v7.0" in content:
                version = "v7.0"
            elif "v6.0" in content:
                version = "v6.0"
            
            # Quick Start         
            has_quickstart = "Quick Start" in content or "     " in content
            
            self.knowledge['readme'] = {
                'version': version,
                'has_quickstart': has_quickstart,
                'raw_content': content,
                'last_read': datetime.now().isoformat()
            }
            
            logger.info(f"  README.md read: v{version}")
            return self.knowledge['readme']
            
        except Exception as e:
            logger.error(f"Error reading README.md: {e}")
            return {}
    
    def scan_current_state(self) -> Dict[str, Any]:
        """
                    
        
        Returns:
                    
        """
        try:
            state = {
                'timestamp': datetime.now().isoformat(),
                'core_structure': {},
                'data_files': {},
                'reports': {}
            }
            
            # Core      
            core_path = self.project_root / "Core"
            if core_path.exists():
                core_dirs = [d.name for d in core_path.iterdir() if d.is_dir()]
                state['core_structure'] = {
                    'exists': True,
                    'subdirs': core_dirs,
                    'count': len(core_dirs)
                }
            
            #          
            data_path = self.project_root / "data"
            if data_path.exists():
                important_files = [
                    'central_registry.json',
                    'data/Memory/memory.db',
                    'cognitive_evaluation.json'
                ]
                state['data_files'] = {
                    f: (data_path / f).exists() 
                    for f in important_files
                }
            
            #             
            reports_path = self.project_root / "reports"
            if reports_path.exists():
                latest_eval = reports_path / "evaluation_latest.json"
                latest_bench = reports_path / "comprehensive_benchmark_latest.json"
                
                state['reports'] = {
                    'evaluation_latest': latest_eval.exists(),
                    'benchmark_latest': latest_bench.exists()
                }
                
                #              
                if latest_bench.exists():
                    try:
                        with open(latest_bench, 'r', encoding='utf-8') as f:
                            bench_data = json.load(f)
                        state['reports']['benchmark_score'] = {
                            'total': bench_data.get('grand_total', 0),
                            'percentage': bench_data.get('percentage', 0),
                            'grade': bench_data.get('grade', 'Unknown')
                        }
                    except:
                        pass
            
            self.knowledge['current_state'] = state
            logger.info("  Current state scanned")
            return state
            
        except Exception as e:
            logger.error(f"Error scanning current state: {e}")
            return {}
    
    def full_self_scan(self) -> Dict[str, Any]:
        """
                 -                 
        
        Returns:
                       
        """
        logger.info("="*70)
        logger.info("  Starting Full Self-Awareness Scan")
        logger.info("="*70)
        
        self.read_codex()
        self.read_architecture()
        self.read_protocols()
        self.read_readme()
        self.scan_current_state()
        
        summary = {
            'scan_time': datetime.now().isoformat(),
            'codex_loaded': bool(self.knowledge['codex']),
            'architecture_loaded': bool(self.knowledge['architecture']),
            'protocols_count': self.knowledge['protocols'].get('count', 0),
            'readme_loaded': bool(self.knowledge['readme']),
            'current_state_scanned': bool(self.knowledge['current_state']),
            'full_knowledge': self.knowledge
        }
        
        logger.info("="*70)
        logger.info("  Self-Awareness Scan Complete")
        logger.info(f"   - CODEX: {summary['codex_loaded']}")
        logger.info(f"   - Architecture: {summary['architecture_loaded']}")
        logger.info(f"   - Protocols: {summary['protocols_count']} documents")
        logger.info(f"   - README: {summary['readme_loaded']}")
        logger.info(f"   - Current State: {summary['current_state_scanned']}")
        logger.info("="*70)
        
        return summary
    
    def suggest_improvements(self) -> List[str]:
        """
                            
        
        Returns:
                    
        """
        suggestions = []
        
        # CODEX      
        if self.knowledge['codex'].get('philosophy', {}).get('no_llm_dependency'):
            suggestions.append(
                "  CODEX   : LLM       -                "
            )
        
        # Architecture      
        arch_pillars = self.knowledge['architecture'].get('pillars', [])
        expected_pillars = ["Foundation", "Intelligence", "Memory", "Interface", "Evolution", "Creativity"]
        missing_pillars = [p for p in expected_pillars if p not in arch_pillars]
        
        if missing_pillars:
            suggestions.append(
                f"   Architecture:     Pillar       - {', '.join(missing_pillars)}"
            )
        
        #            
        state = self.knowledge.get('current_state', {})
        if state:
            #           
            bench_score = state.get('reports', {}).get('benchmark_score', {})
            if bench_score:
                percentage = bench_score.get('percentage', 0)
                if percentage < 85:
                    suggestions.append(
                        f"  Benchmark:    {percentage:.1f}% - S+   (85%+)          "
                    )
                elif percentage >= 90:
                    suggestions.append(
                        f"  Benchmark: {percentage:.1f}% - SSS      !"
                    )
        
        #                  
        suggestions.append(
            "        : instant_internet_sync.py + distributed_consciousness.py       "
        )
        
        if not suggestions:
            suggestions.append("         -             ")
        
        return suggestions

    def introspect_thought(self, thought_history: Any) -> str:
        """
        Explain the causality of a thought based on its trace.
        """
        # Duck typing check for trace
        if not hasattr(thought_history, 'get_narrative'):
            return "I cannot recall the history of this thought."
            
        narrative = thought_history.get_narrative()
        
        # Meta-analysis
        analysis = f"\n[Meta-Cognition]: This thought emerged through {len(thought_history.events)} stages of cognitive processing."
        
        return narrative + analysis
    
    def generate_self_awareness_report(self, output_path: Optional[Path] = None) -> str:
        """
                    
        
        Args:
            output_path:          
        
        Returns:
                     
        """
        if output_path is None:
            output_path = self.project_root / "reports" / "system_self_awareness_report.json"
        
        output_path.parent.mkdir(exist_ok=True)
        
        report = {
            'generated_at': datetime.now().isoformat(),
            'elysia_version': self.knowledge['architecture'].get('version', 'Unknown'),
            'knowledge_base': {
                'codex': {
                    'loaded': bool(self.knowledge['codex']),
                    'laws_count': len(self.knowledge['codex'].get('laws_of_resonance', [])),
                    'cognitive_systems': self.knowledge['codex'].get('cognitive_systems', [])
                },
                'architecture': {
                    'loaded': bool(self.knowledge['architecture']),
                    'version': self.knowledge['architecture'].get('version', 'Unknown'),
                    'pillars': self.knowledge['architecture'].get('pillars', [])
                },
                'protocols': {
                    'loaded': bool(self.knowledge['protocols']),
                    'count': self.knowledge['protocols'].get('count', 0)
                },
                'readme': {
                    'loaded': bool(self.knowledge['readme']),
                    'version': self.knowledge['readme'].get('version', 'Unknown')
                }
            },
            'current_state': self.knowledge.get('current_state', {}),
            'improvement_suggestions': self.suggest_improvements()
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        logger.info(f"  Self-awareness report saved: {output_path}")
        return str(output_path)


def main():
    """      """
    logging.basicConfig(level=logging.INFO)
    
    awareness = SystemSelfAwareness()
    summary = awareness.full_self_scan()
    
    print("\n" + "="*70)
    print("  Improvement Suggestions:")
    print("="*70)
    for suggestion in awareness.suggest_improvements():
        print(f"  {suggestion}")
    print("="*70)
    
    #       
    report_path = awareness.generate_self_awareness_report()
    print(f"\n  Report saved: {report_path}")


if __name__ == "__main__":
    main()
