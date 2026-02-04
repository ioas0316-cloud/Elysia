"""
[Project Elysia] Autonomous Learner
===================================
Phase 500+: Self-Directed Knowledge Acquisition
Elysia autonomously discovers, reads, and internalizes her own codebase.

"아빠가 명령하지 않아도, 스스로 배우는 존재."
"""

import os
import time
import hashlib
from pathlib import Path
from typing import Set, Dict, List
import threading

from Core.S1_Body.L5_Mental.Digestion.knowledge_ingestor import get_knowledge_ingestor
from Core.S1_Body.L5_Mental.Digestion.universal_digestor import get_universal_digestor
from Core.S1_Body.L5_Mental.Digestion.phase_absorber import get_phase_absorber
from Core.S1_Body.L5_Mental.Digestion.entropy_purger import get_entropy_purger
from Core.S1_Body.Tools.Scripts.plasticity_log import plasticity_logger


class AutonomousLearner:
    """
    자율 학습자 (Autonomous Learner)
    Elysia proactively discovers and consumes knowledge without user commands.
    
    "나는 가르침을 기다리지 않는다. 스스로 배운다."
    "나의 세계에는 경계가 없다."
    """
    
    def __init__(self, 
                 root_path: str = r"c:\Elysia",
                 scan_interval: float = 60.0,  # seconds between scans
                 extensions: List[str] = None):
        self.root_path = Path(root_path)
        self.scan_interval = scan_interval
        self.extensions = extensions or ['.md', '.txt', '.py', '.json']
        
        # Dynamic scan paths - Elysia can expand these herself
        self.scan_paths: Set[Path] = {self.root_path}
        
        # External knowledge sources (URLs, APIs, etc.)
        self.external_sources: List[str] = []
        
        # Track what has been consumed
        self.consumed_files: Dict[str, str] = {}  # filepath -> hash
        self.consumed_urls: Set[str] = set()
        self.state_file = self.root_path / "data" / "State" / "learned_files.txt"
        self.paths_file = self.root_path / "data" / "State" / "scan_paths.txt"
        
        self._load_state()
        
        self.running = False
        self._thread = None
        
        # Components
        self.ingestor = get_knowledge_ingestor()
        self.digestor = get_universal_digestor()
        self.absorber = get_phase_absorber()
        self.purger = get_entropy_purger()
    
    def expand_world(self, new_path: str):
        """Expand cognitive boundaries by adding a new scan path."""
        path = Path(new_path)
        if path.exists() and path.is_dir():
            self.scan_paths.add(path)
            self._save_paths()
            # print(f"🌍 [EXPANSION] New world discovered: {new_path}")
            return True
        return False
    
    def add_external_source(self, url: str):
        """Add an external knowledge source (URL)."""
        if url not in self.external_sources:
            self.external_sources.append(url)
            # print(f"🌐 [EXTERNAL] New source added: {url}")
    
    def ingest_url(self, url: str) -> int:
        """Consume knowledge from a URL."""
        if url in self.consumed_urls:
            return 0
        
        try:
            import urllib.request
            with urllib.request.urlopen(url, timeout=10) as response:
                content = response.read().decode('utf-8', errors='ignore')
            
            chunks = self.ingestor.ingest_text(content, source=f"url:{url[:50]}")
            all_nodes = []
            for chunk in chunks:
                nodes = self.digestor.digest(chunk)
                all_nodes.extend(nodes)
            
            absorbed = self.absorber.absorb(all_nodes)
            self.consumed_urls.add(url)
            
            # print(f"🌐 [URL] Consumed: {url[:50]}... ({absorbed} nodes)")
            return absorbed
        except Exception as e:
            return 0
    
    def _load_state(self):
        """Load previously consumed files."""
        if self.state_file.exists():
            with open(self.state_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('|')
                    if len(parts) == 2:
                        self.consumed_files[parts[0]] = parts[1]
    
    def _save_state(self):
        """Persist consumed files state."""
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, 'w', encoding='utf-8') as f:
            for path, hash_val in self.consumed_files.items():
                f.write(f"{path}|{hash_val}\n")
    
    def _save_paths(self):
        """Persist expanded scan paths."""
        self.paths_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.paths_file, 'w', encoding='utf-8') as f:
            for path in self.scan_paths:
                f.write(f"{path}\n")
    
    def _load_paths(self):
        """Load previously expanded scan paths."""
        if self.paths_file.exists():
            with open(self.paths_file, 'r', encoding='utf-8') as f:
                for line in f:
                    path = Path(line.strip())
                    if path.exists():
                        self.scan_paths.add(path)
    
    def _file_hash(self, filepath: Path) -> str:
        """Get hash of file contents."""
        try:
            with open(filepath, 'rb') as f:
                return hashlib.md5(f.read()).hexdigest()[:16]
        except:
            return ""
    
    def discover_new_files(self) -> List[Path]:
        """Find files that haven't been consumed or have changed."""
        new_files = []
        
        # Scan ALL paths in scan_paths (Elysia can expand this herself)
        for scan_root in self.scan_paths:
            for ext in self.extensions:
                try:
                    for filepath in scan_root.rglob(f"*{ext}"):
                        # Skip very large files and some specific patterns
                        try:
                            if filepath.stat().st_size > 100000:  # 100KB limit
                                continue
                        except:
                            continue
                        if '__pycache__' in str(filepath):
                            continue
                        if '.git' in str(filepath):
                            continue
                        if 'node_modules' in str(filepath):
                            continue
                        
                        file_hash = self._file_hash(filepath)
                        str_path = str(filepath)
                        
                        # Check if new or changed
                        if str_path not in self.consumed_files:
                            new_files.append(filepath)
                        elif self.consumed_files[str_path] != file_hash:
                            new_files.append(filepath)  # Changed
                except:
                    continue  # Skip inaccessible paths
        
        return new_files
    
    def consume_file(self, filepath: Path) -> int:
        """Consume a single file through the full digestion pipeline."""
        try:
            # Ingest
            chunks = self.ingestor.ingest_file(str(filepath))
            if not chunks:
                return 0
            
            # Digest
            all_nodes = []
            for chunk in chunks:
                nodes = self.digestor.digest(chunk)
                all_nodes.extend(nodes)
            
            # Absorb
            absorbed = self.absorber.absorb(all_nodes)
            
            # Record consumption
            self.consumed_files[str(filepath)] = self._file_hash(filepath)
            
            # Log the learning event
            plasticity_logger.log_event(
                "LEARN",
                {"file": filepath.name, "nodes": len(all_nodes)},
                len(all_nodes) * 0.001  # Small resonance gain per node
            )
            
            return absorbed
            
        except Exception as e:
            print(f"⚠️ Failed to consume {filepath.name}: {e}")
            return 0
    
    def learning_cycle(self) -> Dict:
        """Run one autonomous learning cycle."""
        # Step 1: Discover new knowledge
        new_files = self.discover_new_files()
        
        if not new_files:
            return {"discovered": 0, "absorbed": 0, "purged": 0}
        
        # Step 2: Consume each file
        total_absorbed = 0
        for filepath in new_files[:5]:  # Limit per cycle to avoid overload
            absorbed = self.consume_file(filepath)
            total_absorbed += absorbed
            # print(f"📖 [AUTO-LEARN] Consumed: {filepath.name} ({absorbed} nodes)")
        
        # Step 3: Save state
        self._save_state()
        
        # Step 4: Periodic purge (every 10 cycles or so)
        purge_stats = {"total_affected": 0}
        if len(self.consumed_files) % 10 == 0:
            purge_stats = self.purger.full_purge_cycle()
        
        return {
            "discovered": len(new_files),
            "absorbed": total_absorbed,
            "purged": purge_stats.get("total_affected", 0)
        }
    
    def start_background(self):
        """Start autonomous learning in background thread."""
        if self.running:
            return
        
        self.running = True
        
        def loop():
            while self.running:
                try:
                    stats = self.learning_cycle()
                    if stats["absorbed"] > 0:
                        # print(f"🧠 [AUTONOMOUS] Learned {stats['absorbed']} new concepts from {stats['discovered']} files")
                        pass
                except Exception as e:
                    print(f"⚠️ [AUTONOMOUS] Learning error: {e}")
                
                time.sleep(self.scan_interval)
        
        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()
        print("🌱 [AUTONOMOUS LEARNER] Active - Elysia is now self-learning.")
    
    def stop(self):
        """Stop background learning."""
        self.running = False
        self._save_state()


# Singleton
_autonomous_learner = None

def get_autonomous_learner() -> AutonomousLearner:
    global _autonomous_learner
    if _autonomous_learner is None:
        _autonomous_learner = AutonomousLearner()
    return _autonomous_learner


if __name__ == "__main__":
    print("🌱 Testing Autonomous Learner...")
    
    learner = get_autonomous_learner()
    
    # Run one cycle
    stats = learner.learning_cycle()
    
    print(f"\n📊 Learning Cycle Results:")
    print(f"   - Discovered: {stats['discovered']} new files")
    print(f"   - Absorbed: {stats['absorbed']} nodes")
    print(f"   - Purged: {stats['purged']} redundant nodes")
    
    print("\n🎉 Autonomous Learner operational!")
