#!/usr/bin/env python3
"""
Demo: Resonance vs Crawling
============================

Interactive demonstration of the paradigm shift from crawling to resonance.

This script shows:
1. Traditional crawling approach (heavy, dead, inefficient)
2. Resonance synchronization approach (light, living, efficient)
3. Side-by-side comparison

Run: python demo_resonance_vs_crawling.py
"""

import sys
import os
import time
import logging

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Core.Integration.resonance_data_connector import ResonanceDataConnector

# Set up logging
logging.basicConfig(
    level=logging.WARNING,  # Only show warnings and errors for cleaner demo
    format='%(message)s'
)

def print_header(text):
    """Print a fancy header."""
    print("\n" + "="*80)
    print(f"  {text}")
    print("="*80 + "\n")

def print_section(text):
    """Print a section divider."""
    print("\n" + "-"*80)
    print(f"  {text}")
    print("-"*80 + "\n")

def simulate_traditional_crawling(concepts):
    """Simulate traditional web crawling approach."""
    print_section("Traditional Approach: Crawling (크롤링)")
    
    print("📚 Method: Download entire pages from the internet")
    print("   - Fetch full HTML content")
    print("   - Parse and extract text")
    print("   - Store everything in database")
    print()
    
    total_bandwidth = 0
    total_storage = 0
    total_time = 0
    
    print("Processing concepts:")
    for i, concept in enumerate(concepts, 1):
        # Simulate crawling delay
        time.sleep(0.3)  # Simulate network delay
        
        bandwidth = 100_000  # 100KB per concept (typical Wikipedia page)
        storage = 100_000    # Store full text
        crawl_time = 0.5     # Average crawl time
        
        total_bandwidth += bandwidth
        total_storage += storage
        total_time += crawl_time
        
        print(f"   [{i}/{len(concepts)}] Crawling '{concept}'...")
        print(f"       Downloaded: {bandwidth/1000:.1f} KB")
        print(f"       Stored: {storage/1000:.1f} KB")
        print(f"       Time: {crawl_time:.1f}s")
    
    print()
    print("❌ Traditional Crawling Results:")
    print(f"   Total bandwidth used: {total_bandwidth/1000:.1f} KB ({total_bandwidth/1_000_000:.2f} MB)")
    print(f"   Total storage needed: {total_storage/1000:.1f} KB ({total_storage/1_000_000:.2f} MB)")
    print(f"   Total time taken: {total_time:.1f} seconds")
    print(f"   Status: Static (will become outdated)")
    print(f"   Copyright: Issues (possessing full copies)")
    print()
    
    return {
        "bandwidth": total_bandwidth,
        "storage": total_storage,
        "time": total_time,
        "freshness": "static"
    }

def demonstrate_resonance_sync(concepts):
    """Demonstrate resonance synchronization approach."""
    print_section("Resonance Approach: Synchronization (동기화)")
    
    print("🌊 Method: Resonate with essence and extract Pattern DNA")
    print("   - Probe for core meaning (like tasting with tongue)")
    print("   - Extract Pattern DNA seed")
    print("   - Establish live sync channel")
    print()
    
    connector = ResonanceDataConnector()
    
    print("Processing concepts:")
    start_time = time.time()
    
    results = []
    for i, concept in enumerate(concepts, 1):
        result = connector.resonate_with_concept(concept)
        results.append(result)
        
        print(f"   [{i}/{len(concepts)}] Resonating with '{concept}'...")
        print(f"       Seed size: {result['seed_size']/1000:.2f} KB")
        print(f"       Bandwidth saved: {result['bandwidth_saved']} bytes")
        print(f"       Compression: {result['compression_ratio']:.1f}x")
    
    elapsed_time = time.time() - start_time
    
    # Calculate totals
    total_seed_size = sum(r['seed_size'] for r in results if r['success'])
    total_bandwidth_saved = sum(r['bandwidth_saved'] for r in results if r['success'])
    
    print()
    print("✅ Resonance Synchronization Results:")
    print(f"   Total seed storage: {total_seed_size/1000:.2f} KB")
    print(f"   Total bandwidth used: {total_seed_size/1000:.2f} KB")
    print(f"   Total time taken: {elapsed_time:.2f} seconds")
    print(f"   Status: Live (always synchronized)")
    print(f"   Copyright: Clean (accessing, not possessing)")
    print()
    
    return {
        "bandwidth": total_seed_size,
        "storage": total_seed_size,
        "time": elapsed_time,
        "freshness": "live",
        "connector": connector
    }

def show_comparison(traditional, resonance):
    """Show side-by-side comparison."""
    print_section("Performance Comparison")
    
    bandwidth_saved = traditional["bandwidth"] - resonance["bandwidth"]
    bandwidth_saved_percent = (bandwidth_saved / traditional["bandwidth"]) * 100
    
    storage_saved = traditional["storage"] - resonance["storage"]
    storage_saved_percent = (storage_saved / traditional["storage"]) * 100
    
    time_saved = traditional["time"] - resonance["time"]
    speedup = traditional["time"] / resonance["time"] if resonance["time"] > 0 else float('inf')
    
    print("📊 Metrics Comparison:\n")
    
    print(f"   Bandwidth:")
    print(f"      Traditional: {traditional['bandwidth']/1000:.1f} KB")
    print(f"      Resonance:   {resonance['bandwidth']/1000:.2f} KB")
    print(f"      Saved:       {bandwidth_saved/1000:.1f} KB ({bandwidth_saved_percent:.1f}%)")
    print()
    
    print(f"   Storage:")
    print(f"      Traditional: {traditional['storage']/1000:.1f} KB")
    print(f"      Resonance:   {resonance['storage']/1000:.2f} KB")
    print(f"      Saved:       {storage_saved/1000:.1f} KB ({storage_saved_percent:.1f}%)")
    print()
    
    print(f"   Time:")
    print(f"      Traditional: {traditional['time']:.1f} seconds")
    print(f"      Resonance:   {resonance['time']:.2f} seconds")
    print(f"      Saved:       {time_saved:.1f} seconds (speedup: {speedup:.1f}x)")
    print()
    
    print(f"   Freshness:")
    print(f"      Traditional: {traditional['freshness']} (becomes outdated)")
    print(f"      Resonance:   {resonance['freshness']} (always current)")
    print()
    
    print("💡 Key Advantages of Resonance:")
    print(f"   ✅ {bandwidth_saved_percent:.1f}% less bandwidth")
    print(f"   ✅ {storage_saved_percent:.1f}% less storage")
    print(f"   ✅ {speedup:.1f}x faster")
    print(f"   ✅ Real-time synchronization")
    print(f"   ✅ Copyright-friendly")
    print()

def demonstrate_live_retrieval(connector, concept):
    """Demonstrate retrieving knowledge at different resolutions."""
    print_section("Live Knowledge Retrieval")
    
    print(f"🔍 Retrieving '{concept}' at different resolutions...")
    print("   (Same seed generates different detail levels)\n")
    
    # Low resolution
    print("   Resolution: 50 (low detail)")
    knowledge_low = connector.retrieve_knowledge(concept, resolution=50)
    if knowledge_low:
        harmonics = len(knowledge_low['knowledge'].get('waveform', []))
        print(f"   Generated: {harmonics} harmonics")
    
    # Medium resolution
    print("\n   Resolution: 100 (medium detail)")
    knowledge_med = connector.retrieve_knowledge(concept, resolution=100)
    if knowledge_med:
        harmonics = len(knowledge_med['knowledge'].get('waveform', []))
        print(f"   Generated: {harmonics} harmonics")
    
    # High resolution
    print("\n   Resolution: 200 (high detail)")
    knowledge_high = connector.retrieve_knowledge(concept, resolution=200)
    if knowledge_high:
        harmonics = len(knowledge_high['knowledge'].get('waveform', []))
        print(f"   Generated: {harmonics} harmonics")
    
    print("\n   ✅ Same seed → Different resolutions (infinite scalability!)")
    print()

def show_philosophy():
    """Display the philosophical meaning."""
    print_section("Philosophy")
    
    print("🌊 만류귀종(萬流歸宗) - All Streams Return to One Source\n")
    
    print("   Traditional Approach (Crawling):")
    print("   ❌ \"남들은 바닷물을 다 퍼 마셔야 소금맛을 안다\"")
    print("      (Others must drink the entire ocean to taste the salt)")
    print()
    print("   Resonance Approach (Synchronization):")
    print("   ✅ \"우리는 혀끝만 살짝 대고도 '아, 짜다!' 하고 공명한다\"")
    print("      (We just touch our tongue and resonate: 'Ah, salty!')")
    print()
    
    print("   Key Insights:")
    print("   • 수집가는 무겁고, 여행자는 가볍습니다")
    print("     (Collectors are heavy, travelers are light)")
    print()
    print("   • 데이터를 '소유'하지 말고 '접속'하라")
    print("     (Access data, don't possess it)")
    print()
    print("   • 우린 그냥 동기화하면 됩니다")
    print("     (We just synchronize)")
    print()

def main():
    """Run the complete demonstration."""
    print_header("🌊 RESONANCE vs CRAWLING: A Paradigm Shift Demo")
    
    print("This demo compares two approaches to knowledge acquisition:")
    print("  1. Traditional Crawling (크롤링) - Heavy, Dead, Inefficient")
    print("  2. Resonance Synchronization (동기화) - Light, Living, Efficient")
    print()
    
    # Sample concepts
    concepts = ["Love", "Peace", "Harmony", "Wisdom", "Light"]
    
    print(f"📚 Sample concepts to learn: {', '.join(concepts)}")
    print()
    
    input("Press Enter to start the traditional crawling approach...")
    
    # Demonstrate traditional crawling
    traditional_results = simulate_traditional_crawling(concepts)
    
    input("\nPress Enter to start the resonance synchronization approach...")
    
    # Demonstrate resonance
    resonance_results = demonstrate_resonance_sync(concepts)
    
    # Show comparison
    show_comparison(traditional_results, resonance_results)
    
    # Demonstrate live retrieval
    input("Press Enter to demonstrate live knowledge retrieval...")
    demonstrate_live_retrieval(resonance_results['connector'], concepts[0])
    
    # Show philosophy
    show_philosophy()
    
    print_header("Demo Complete")
    print("✨ 오늘 밤도 가볍고 우아하게, Tune in! 🎧✨🌍")
    print("   (Tonight, light and elegant, tune in!)")
    print()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✋ Demo interrupted by user.")
        print("   수집가는 무겁고, 여행자는 가볍습니다!")
        sys.exit(0)
