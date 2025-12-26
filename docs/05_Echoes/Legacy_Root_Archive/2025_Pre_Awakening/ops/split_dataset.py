"""
Dataset Splitter (The Chewing Mechanism)
========================================
Reads the massive compressed BZ2 dump ONCE and splits it into
digestible raw JSONL shards for high-speed parallel ingestion.

Target: kowiki-latest-pages-articles.xml.bz2
Output: data/shards/shard_001.jsonl, shard_002.jsonl...
"""

import os
import sys
import json
import time
from pathlib import Path

sys.path.append(r'c:\Elysia')
from Core.Autonomy.wikipedia_dump_parser import WikipediaDumpParser

def split_dataset():
    dump_path = "c:\\Elysia\\data\\wikipedia\\kowiki-latest-pages-articles.xml.bz2"
    output_dir = Path("c:\\Elysia\\data\\shards")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    parser = WikipediaDumpParser(dump_path)
    
    current_shard_idx = 1
    current_shard_size = 0
    max_shard_size = 50 * 1024 * 1024 # 50MB per shard
    
    current_file_path = output_dir / f"shard_{current_shard_idx:03d}.jsonl"
    current_handle = open(current_file_path, 'w', encoding='utf-8')
    
    print(f"\n🔪 Operation Chewing initiated...")
    print(f"   Source: {dump_path}")
    print(f"   Target: {output_dir}")
    print("---------------------------------------------")
    
    count = 0
    start_time = time.time()
    
    try:
        for article in parser.stream_articles(max_articles=None, min_length=50):
            # Serialize
            line = json.dumps(article, ensure_ascii=False) + "\n"
            line_bytes = len(line.encode('utf-8'))
            
            # Check rotation
            if current_shard_size + line_bytes > max_shard_size:
                current_handle.close()
                print(f"   📦 Sealed {current_file_path.name} ({current_shard_size/1024/1024:.2f} MB)")
                
                current_shard_idx += 1
                current_shard_size = 0
                current_file_path = output_dir / f"shard_{current_shard_idx:03d}.jsonl"
                current_handle = open(current_file_path, 'w', encoding='utf-8')
            
            current_handle.write(line)
            current_shard_size += line_bytes
            count += 1
            
            if count % 1000 == 0:
                print(f"\r   🥩 Chewed {count} articles... (Current Shard: {current_shard_size/1024/1024:.1f} MB)", end="")
                
    except KeyboardInterrupt:
        print("\n🛑 Chewing Interrupted.")
    except Exception as e:
        print(f"\n❌ Error: {e}")
    finally:
        current_handle.close()
        print(f"\n✅ Operation Complete. Total Articles: {count}")

if __name__ == "__main__":
    split_dataset()
