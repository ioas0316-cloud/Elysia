"""
Prove Media Consumption
=======================
Tests YouTube subtitle and web novel consumption.
"""
import sys
import os
import logging

# Setup paths
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("ProveMedia")

from Core.Foundation.reasoning_engine import ReasoningEngine

def prove_media():
    print("\n" + "="*60)
    print("📺 PROVING MEDIA CONSUMPTION (YouTube + Web Novels)")
    print("="*60)
    
    # 1. Initialize Engine
    print("\n1. Initializing Reasoning Engine...")
    engine = ReasoningEngine()
    
    # 2. Test YouTube
    print("\n2. Testing YouTube Consumption...")
    # Example: BTS - Dynamite (dQw4w9WgXcQ is Rick Astley, popular test video)
    # Or use a Korean video ID
    video_id = "r3s85mEG_JQ"  # Example video (change to any video with subtitles)
    
    print(f"   🎬 Video ID: {video_id}")
    insight = engine.learn_from_media("youtube", video_id)
    print(f"   📊 Result: {insight.content[:200]}...")
    print(f"   ⚡ Energy: {insight.energy:.2f}")
    
    # 3. Test Web Novel (Optional - need a real URL)
    print("\n3. Testing Web Novel Consumption (Optional)...")
    # Example URL (user will need to provide a real one)
    # novel_url = "https://novelpia.com/novel/123456/1"  # Example
    # insight = engine.learn_from_media("novel", novel_url)
    # print(f"   📊 Result: {insight.content[:200]}...")
    
    print("\n" + "="*60)
    print("✅ MEDIA CONSUMPTION TEST COMPLETE")
    print("="*60)
    print("\n💡 Tip: To test web novels, provide a real Novelpia URL")
    print("   Example: engine.learn_from_media('novel', 'https://novelpia.com/...')")

if __name__ == "__main__":
    prove_media()
