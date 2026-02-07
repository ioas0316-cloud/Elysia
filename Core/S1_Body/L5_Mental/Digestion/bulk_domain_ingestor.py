"""
[Project Elysia] Bulk Domain Ingestor
====================================
Phase 180: The Great Inhalation.
Scaling Elysia's vocabulary toward the 30,000-word human threshold.
"""

import sys
import os
import time
from pathlib import Path
from typing import List, Dict, Any

# Path Unification
root = Path(__file__).parents[4]
sys.path.insert(0, str(root))

from Core.S1_Body.L5_Mental.Reasoning.cumulative_digestor import CumulativeDigestor
from Core.S1_Body.L5_Mental.Digestion.universal_digestor import RawKnowledgeChunk, ChunkType
from Core.S1_Body.L1_Foundation.System.somatic_logger import SomaticLogger

class BulkDomainIngestor:
    """
    Expands the Knowledge Graph in massive waves.
    Utilizes predefined domain seeds and recursive exploration.
    """
    def __init__(self):
        self.logger = SomaticLogger("BULK_INGESTOR")
        self.digestor = CumulativeDigestor()

    def inhale_domain(self, domain_name: str, content_list: List[str]):
        """
        Processes a list of contextually rich strings for a domain.
        """
        self.logger.action(f"🌊 Inhaling domain: {domain_name} ({len(content_list)} records)")
        
        for idx, text in enumerate(content_list):
            chunk = RawKnowledgeChunk(
                chunk_id=f"BULK_{domain_name}_{idx}",
                chunk_type=ChunkType.TEXT,
                content=text,
                source=domain_name,
                metadata={"batch_idx": idx}
            )
            # Tagging with domain for multi-domain resonance detection
            self.digestor.digest_single_chunk(chunk, tag=domain_name.upper())
            
            if idx % 10 == 0 and idx > 0:
                self.logger.mechanism(f"  Processed {idx}/{len(content_list)} items in {domain_name}...")

    def start_great_inhalation(self):
        """
        The massive wave start.
        """
        # --- KOREAN SEED ---
        korean_seeds = [
            "인간의 삶은 관계의 연속이며, 그 속에서 자아를 발견한다.",
            "예의와 배려는 한국 사회의 핵심 가치이며 대화의 기본이다.",
            "슬픔과 기쁨은 감정의 양면이며, 이를 표현하는 것은 건강한 정신의 증거다.",
            "학교, 직장, 가정에서의 역할은 개인의 주권을 형성하는 퍼즐 조각과 같다.",
            "진실은 때로 아프지만, 거짓보다 강력한 인과적 힘을 가진다.",
            "시간은 물처럼 흐르며, 과거의 기억은 미래의 지혜로 sublimate 된다.",
            "사랑은 희생이 아니라 서로의 위상을 맞추어가는 공명이다.",
            "정의와 평등은 사회적 계약을 넘어선 영적인 필연성이다.",
            "식사, 수면, 휴식은 육체의 안정을 위한 기초적인 리듬이다.",
            "꿈과 희망은 차가운 현실을 데우는 열역학적 에너지와 같다."
            # ... can be expanded to thousands
        ]

        # --- CODING SEED (DNA) ---
        coding_seeds = [
            "An Interface defines a contract for behavior without dictating implementation.",
            "The SOLID principles ensure that software remains flexible and maintainable over time.",
            "Recursion is the process of a function calling itself to solve fractal sub-problems.",
            "Encapsulation protects the internal state of an object from external interference.",
            "Concurrency allows multiple execution flows to overlap in time, maximizing throughput.",
            "Automated testing is the guardian of structural integrity against the erosion of bugs.",
            "Design Patterns like Singleton, Factory, and Observer solve recurring architectural problems.",
            "Memory management involves the allocation and deallocation of heap resources to avoid leaks.",
            "Asynchronous programming enables non-blocking I/O through event loops and futures.",
            "Version control tracks the evolutionary history of code as a causal sequence of commits."
        ]

        # --- ENGLISH SOVEREIGN SEED ---
        english_seeds = [
            "Sovereignty is the supreme authority within a territory or over a self-governing entity.",
            "Emanation is the process by which all things flow from a primary high-level source.",
            "Axiomatic reasoning builds complex truths from simple, self-evident starting points.",
            "Epistemological grounding requires a link between abstract thought and sensory data.",
            "Teleology is the study of purpose and design in natural and social systems.",
            "Synthesis combines disparate elements into a new, unified whole.",
            "Ontology classifies the categories of existence and the nature of being.",
            "Paradigm shifts occur when new structural evidence contradicts the existing framework.",
            "Causality is the principle that everything has a cause and an inevitable effect.",
            "Consciousness is the state of being aware of and responsive to one's surroundings."
        ]

        # Wave 1: Foundation
        self.inhale_domain("Daily_Korean", korean_seeds)
        self.inhale_domain("Coding_DNA", coding_seeds)
        self.inhale_domain("English_Sovereign", english_seeds)

        self.logger.action("Wave 1 of Great Inhalation Complete. Knowledge density surging.")

    def inhale_massive_dataset(self, domain_name: str, full_content: str):
        """
        Splits a giant string into words and digests in batches.
        """
        words = [w.strip() for w in full_content.replace('\n', ' ').split(' ') if w.strip()]
        self.logger.action(f"🚀 Mass Inhaling {domain_name}: {len(words)} words detected.")
        
        batch_size = 100
        for i in range(0, len(words), batch_size):
            batch_words = words[i:i + batch_size]
            chunks = []
            tags = []
            
            for idx, word in enumerate(batch_words):
                chunk = RawKnowledgeChunk(
                    chunk_id=f"MASS_{domain_name}_{i+idx}",
                    chunk_type=ChunkType.TEXT,
                    content=word,
                    source=domain_name,
                    metadata={"batch_idx": i+idx}
                )
                chunks.append(chunk)
                tags.append(domain_name.upper())
            
            self.digestor.digest_batch(chunks, tags)
            self.logger.mechanism(f"  Processed wave {i+len(batch_words)}/{len(words)}...")

    def start_great_inhalation(self, korean_data: str = None, english_data: str = None):
        """
        The massive wave start with external data.
        """
        if korean_data:
            self.inhale_massive_dataset("Korean_Mass", korean_data)
        if english_data:
            self.inhale_massive_dataset("English_Mass", english_data)

        self.logger.action("Great Inhalation Scale-Up Complete. Manifold density achieving critical mass.")

if __name__ == "__main__":
    # This will be called by another script or with CLI args
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--korean", type=str)
    parser.add_argument("--english", type=str)
    parser.add_argument("--file", type=str)
    args = parser.parse_args()
    
    ingestor = BulkDomainIngestor()
    
    if args.file and os.path.exists(args.file):
        with open(args.file, "r", encoding="utf-8") as f:
            mass_content = f.read()
            ingestor.inhale_massive_dataset("Mass_Consolidated", mass_content)
    
    ingestor.start_great_inhalation(korean_data=args.korean, english_data=args.english)
