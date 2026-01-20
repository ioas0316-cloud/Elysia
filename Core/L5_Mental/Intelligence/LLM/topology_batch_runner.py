
"""
LLM Topology Batch Runner (거대 모델용)
=====================================
Core.Intelligence.LLM.topology_batch_runner

Qwen2-72B와 같이 여러 파일로 쪼개진(Sharded) 모델을
순차적으로 분석하고 결과를 통합하는 배치 러너입니다.
"""

import os
import sys
import glob
import logging
from collections import defaultdict
import torch
from topology_tracer import get_topology_tracer, ThoughtCircuit

# 로깅 설정
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("BatchRunner")

def run_batch_analysis(model_dir: str):
    """
    디렉토리 내의 모든 safetensors 파일을 분석하여 통합 리포트 생성
    """
    logger.info(f"🚀 Starting batch analysis for: {model_dir}")
    
    # safetensors 파일 찾기
    files = glob.glob(os.path.join(model_dir, "*.safetensors"))
    files.sort()
    
    if not files:
        logger.error("❌ No .safetensors files found!")
        return
        
    logger.info(f"📂 Found {len(files)} shards.")
    
    tracer = get_topology_tracer(threshold=0.01) # 민감도 설정
    
    # 글로벌 통계
    global_stats = {
        "total_params": 0,
        "strong_connections": 0,
        "layers_analyzed": 0,
        "connection_types": defaultdict(int)
    }
    
    global_connection_counts = defaultdict(int)
    
    for i, file_path in enumerate(files):
        filename = os.path.basename(file_path)
        logger.info(f"[{i+1}/{len(files)}] 🕵️ Analyzing {filename}...")
        
        try:
            # 개별 파일 분석
            circuit = tracer.trace(file_path)
            
            # 통계 합산
            global_stats["total_params"] += circuit.total_params
            global_stats["strong_connections"] += circuit.strong_connections
            global_stats["layers_analyzed"] += circuit.layers_analyzed
            
            # 연결 타입 합산
            for conn in circuit.connections:
                global_stats["connection_types"][conn.connection_type] += 1
                
                # 허브 뉴런 카운팅 (소스, 타겟 모두)
                global_connection_counts[conn.source] += 1
                global_connection_counts[conn.target] += 1
                
            # 메모리 정리 (리스트 비우기)
            del circuit
            
        except Exception as e:
            logger.error(f"⚠️ Error analyzing {filename}: {e}")
            
    # 전체 허브 뉴런 계산
    logger.info("🧮 Calculating global hub neurons...")
    sorted_neurons = sorted(global_connection_counts.items(), key=lambda x: -x[1])
    top_hubs = [n for n, count in sorted_neurons[:20]]
    
    print("\n" + "="*60)
    print(f"GIANT MODEL ANATOMY REPORT: {os.path.basename(model_dir)}")
    print("="*60)
    print(f"   Shards Processed: {len(files)}")
    print(f"   Total Parameters: {global_stats['total_params']:,}")
    print(f"   Layers Analyzed: {global_stats['layers_analyzed']}")
    print(f"   Strong Connections: {global_stats['strong_connections']:,}")
    print(f"   Connection Types: {dict(global_stats['connection_types'])}")
    print("-" * 60)
    print(f"   Top 20 Global Hub Neurons (The Elders):")
    print(f"      {top_hubs}")
    print("="*60)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python topology_batch_runner.py <model_directory>")
        sys.exit(1)
        
    run_batch_analysis(sys.argv[1])
