"""
Wave Stream Receiver - P4.0
Receives wave patterns from multiple knowledge sources simultaneously
"""

import asyncio
import logging
from typing import List, AsyncGenerator, Optional
from collections import deque

logger = logging.getLogger(__name__)


class WaveBuffer:
    """Wave buffer for temporary storage"""
    
    def __init__(self, max_size=1000):
        self.buffer = deque(maxlen=max_size)
        self.lock = asyncio.Lock()
    
    async def add(self, wave_pattern):
        """Add wave pattern to buffer"""
        async with self.lock:
            self.buffer.append(wave_pattern)
    
    async def get(self):
        """Get wave pattern from buffer"""
        async with self.lock:
            if self.buffer:
                return self.buffer.popleft()
            return None
    
    def size(self):
        """Get current buffer size"""
        return len(self.buffer)


class WaveStreamReceiver:
    """
    Wave Stream Receiver - 빛을 받듯이 파동 정보 수신
    
    Receives wave patterns from multiple sources simultaneously
    like receiving light from different directions.
    """
    
    def __init__(self):
        self.stream_sources = []
        self.wave_buffer = WaveBuffer(max_size=1000)
        self.running = False
        self.stats = {
            'received': 0,
            'processed': 0,
            'errors': 0
        }
        logger.info("🌊 WaveStreamReceiver initialized")
    
    def add_stream_source(self, source):
        """Add a stream source"""
        self.stream_sources.append(source)
        logger.info(f"Added stream source: {source.__class__.__name__}")
    
    async def receive_streams(self):
        """Receive from all sources simultaneously"""
        self.running = True
        logger.info(f"🌊 Starting wave reception from {len(self.stream_sources)} sources...")
        
        # Create tasks for all sources
        tasks = [
            self.receive_from_source(source)
            for source in self.stream_sources
        ]
        
        # Run all simultaneously
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def receive_from_source(self, source):
        """Receive from a single source continuously"""
        source_name = source.__class__.__name__
        logger.info(f"  📡 Receiving from {source_name}...")
        
        try:
            async for wave_data in source.stream():
                if not self.running:
                    break
                
                # Convert to wave pattern
                wave_pattern = self.to_wave_pattern(wave_data, source_name)
                
                # Add to buffer
                await self.wave_buffer.add(wave_pattern)
                
                self.stats['received'] += 1
                
                if self.stats['received'] % 100 == 0:
                    logger.debug(f"  Received {self.stats['received']} waves, buffer: {self.wave_buffer.size()}")
        
        except Exception as e:
            logger.error(f"Error receiving from {source_name}: {e}")
            self.stats['errors'] += 1
    
    def to_wave_pattern(self, raw_data, source_name):
        """Convert raw data to wave pattern"""
        # Simple conversion for now
        # In full implementation, this would extract frequency, phase, amplitude
        return {
            'data': raw_data,
            'source': source_name,
            'timestamp': asyncio.get_event_loop().time()
        }
    
    async def get_wave(self):
        """Get next wave from buffer"""
        return await self.wave_buffer.get()
    
    def stop(self):
        """Stop receiving"""
        self.running = False
        logger.info("🛑 Stopped wave reception")
    
    def get_stats(self):
        """Get statistics"""
        return {
            **self.stats,
            'buffer_size': self.wave_buffer.size(),
            'sources': len(self.stream_sources)
        }
