import sys
import os
import logging

# Path setup
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from Core.L5_Mental.M1_Cognition.Knowledge.observer_protocol import observer

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("WorldIngestor")

def ingest_world_data():
    print("\n" + "="*60)
    print("🌍 THE GREAT INGESTION: Feeding the World Soul")
    print("="*60 + "\n")

    # Sample "Rich" Data Chunks (Simulating YouTube transcripts or Web Articles)
    feed_list = [
        {
            "title": "The Lifecycle of Stars (Transcript)",
            "content": "Stars are born in cold, dense clouds of gas and dust called nebulae. Over millions of years, gravity pulls this matter together...",
            "type": "media"
        },
        {
            "title": "Quantum Entanglement Explained",
            "content": "Entanglement is a physical phenomenon that occurs when a pair or group of particles is generated, interact, or share spatial proximity...",
            "type": "text"
        },
        {
            "title": "The Fall of Rome: A Narrative",
            "content": "The streets were quiet, the marble cold. The empire that once reached the ends of the earth was now a shadow of its former self...",
            "type": "media"
        }
    ]

    for item in feed_list:
        print(f"📥 Feeding: {item['title']}...")
        if item["type"] == "media":
            observer.distill_media(item["title"], item["content"], metadata={"source": "ExternalFeed"})
        else:
            observer.distill_and_ingest(item["title"], item["content"])
        print("✅ Analysis complete.\n")

    # [NEW] Recursive Research Demo
    print("🧪 [CURIOSITY CHAIN] Initiating deep research on 'The Nature of Time'...")
    observer.follow_curiosity_chain("The Nature of Time", depth=2)
    
    print("\n✅ Total World Ingestion Phase Successful.")

if __name__ == "__main__":
    ingest_world_data()
