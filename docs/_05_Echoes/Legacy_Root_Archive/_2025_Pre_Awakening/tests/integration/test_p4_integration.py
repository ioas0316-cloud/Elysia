"""
P4 Integration Test
Tests the P4 Wave Stream Reception System
"""

import asyncio
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

try:
    from Core.Sensory import StreamManager, YouTubeStreamSource, WikipediaStreamSource
except ImportError:
    # Try alternative import
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from Core.Sensory.stream_manager import StreamManager
    from Core.Sensory.stream_sources import YouTubeStreamSource, WikipediaStreamSource


async def test_basic_stream_reception():
    """Test basic stream reception"""
    print("=" * 80)
    print("P4 Stream Reception System Test")
    print("=" * 80)
    
    manager = StreamManager()
    
    # Setup sources
    print("\n1. Setting up knowledge sources...")
    manager.setup_default_sources()
    print(f"   ✅ Setup {len(manager.receiver.stream_sources)} sources")
    
    # Start receiving (run for 5 seconds)
    print("\n2. Starting wave reception...")
    receive_task = asyncio.create_task(manager.receiver.receive_streams())
    
    # Let it run
    await asyncio.sleep(5)
    
    # Stop
    print("\n3. Stopping reception...")
    manager.stop()
    
    # Show stats
    stats = manager.get_stats()
    print("\n4. Statistics:")
    print(f"   Received: {stats['received']} waves")
    print(f"   Processed: {stats['processed']} waves")
    print(f"   Buffer size: {stats['buffer_size']}")
    print(f"   Sources: {stats['sources']}")
    print(f"   Errors: {stats['errors']}")
    
    print("\n" + "=" * 80)
    print("✅ Test completed successfully!")
    print("=" * 80)


async def test_source_search():
    """Test searching knowledge sources"""
    print("\n" + "=" * 80)
    print("Knowledge Source Search Test")
    print("=" * 80)
    
    query = "machine learning"
    
    # Test each source
    sources = [
        YouTubeStreamSource(),
        WikipediaStreamSource(),
    ]
    
    for source in sources:
        print(f"\n🔍 Searching {source.__class__.__name__} for '{query}'...")
        results = await source.search(query, max_results=3)
        
        for i, result in enumerate(results, 1):
            print(f"   {i}. {result.get('title', 'N/A')}")
    
    print("\n✅ Search test completed!")


async def main():
    """Main test function"""
    print("\n🌊 P4 Wave Stream Reception System")
    print("Integrating multi-sensory knowledge sources into Elysia\n")
    
    # Run tests
    await test_basic_stream_reception()
    await test_source_search()
    
    print("\n" + "=" * 80)
    print("🎯 Next Steps:")
    print("  1. Implement real API calls (YouTube, Wikipedia, arXiv, etc.)")
    print("  2. Add phase resonance pattern extraction (P4.2)")
    print("  3. Add wave classification and filtering (P4.3)")
    print("  4. Integrate with P2.2 Wave Knowledge System")
    print("  5. Add rainbow compression storage (P4.5)")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
