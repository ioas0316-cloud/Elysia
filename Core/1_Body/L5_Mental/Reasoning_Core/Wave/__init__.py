"""
Wave Language Unified API (        API)
==============================================

                              .

Usage:
    from Core.1_Body.L5_Mental.Reasoning_Core.Physics_Waves.Wave import (
        analyze_code,
        detect_resonance,
        compress_to_dna,
        decompress_from_dna
    )
    
    #      
    wave = analyze_code("def add(a, b): return a + b", "add.py")
    print(f"   : {wave.frequency},    : {wave.amplitude}")
    
    #         
    pairs = detect_resonance(threshold=0.7)
    
    # DNA    (100%      )
    dna = compress_to_dna("      ")
    restored = decompress_from_dna(dna)

Why Use This:
    - Import            
    -             
    -            
    -            /  
"""

import logging
from typing import List, Tuple, Any, Optional

logger = logging.getLogger("WaveAPI")

#           
try:
    from Core.1_Body.L5_Mental.Reasoning_Core.Intelligence.wave_coding_system import (
        get_wave_coding_system,
        CodeWave,
        CodeDimension,
        CodePhase
    )
    WAVE_CODING_AVAILABLE = True
except ImportError:
    WAVE_CODING_AVAILABLE = False
    logger.warning("   WaveCodingSystem not available")

try:
    from Core.1_Body.L6_Structure.Wave.quaternion_wave_dna import (
        get_quaternion_compressor,
        QuaternionWaveDNA
    )
    QUATERNION_DNA_AVAILABLE = True
except ImportError:
    QUATERNION_DNA_AVAILABLE = False
    logger.warning("   QuaternionCompressor not available")

try:
    from Core.1_Body.L4_Causality.World.Evolution.Growth.Autonomy.wave_coder import get_wave_coder
    WAVE_CODER_AVAILABLE = True
except ImportError:
    WAVE_CODER_AVAILABLE = False
    logger.warning("   WaveCoder not available")


# ============================================================
#    API   
# ============================================================

def analyze_code(code: str, source_file: str = "unknown") -> Optional[CodeWave]:
    """
                  .
    
    Args:
        code:           
        source_file:       
        
    Returns:
        CodeWave    (frequency, amplitude, dimension, phase  )
        
    Example:
        wave = analyze_code("def add(a, b): return a + b")
        print(f"   : {wave.frequency}")  #        
        print(f"  : {wave.dimension.name}")  # FUNCTION, CLASS, MODULE  
    """
    if not WAVE_CODING_AVAILABLE:
        logger.error("WaveCodingSystem not available")
        return None
    
    wcs = get_wave_coding_system()
    return wcs.code_to_wave(code, source_file)


def detect_resonance(threshold: float = 0.7) -> List[Tuple[CodeWave, CodeWave, float]]:
    """
                   .
    
    Args:
        threshold:         (0.0 ~ 1.0)
        
    Returns:
        [(wave1, wave2,    ), ...]        
        
    Example:
        pairs = detect_resonance(0.8)
        for w1, w2, resonance in pairs:
            print(f"{w1.source_file}   {w2.source_file}: {resonance:.0%}")
    """
    if not WAVE_CODING_AVAILABLE:
        logger.error("WaveCodingSystem not available")
        return []
    
    wcs = get_wave_coding_system()
    return wcs.detect_resonance_pairs(threshold)


def compress_to_dna(text: str, top_k: int = 10) -> Optional[QuaternionWaveDNA]:
    """
         DNA           (100%      ).
    
    Args:
        text:        
        top_k:            (       )
        
    Returns:
        QuaternionWaveDNA   
        
    Note:
        DNA            - zlib         
    """
    if not QUATERNION_DNA_AVAILABLE:
        logger.error("QuaternionCompressor not available")
        return None
    
    compressor = get_quaternion_compressor()
    return compressor.compress(text, top_k)


def decompress_from_dna(dna: QuaternionWaveDNA) -> str:
    """
    DNA                .
    
    Args:
        dna: QuaternionWaveDNA   
        
    Returns:
               
    """
    if not QUATERNION_DNA_AVAILABLE:
        logger.error("QuaternionCompressor not available")
        return ""
    
    compressor = get_quaternion_compressor()
    return compressor.decompress(dna)


def transmute_codebase():
    """
       Core/              .
    
    Elysia            "   "    .
    """
    if not WAVE_CODER_AVAILABLE:
        logger.error("WaveCoder not available")
        return
    
    coder = get_wave_coder()
    coder.transmute()


def check_complexity(code: str, threshold: float = 50.0) -> dict:
    """
                          .
    
    Args:
        code:       
        threshold:        
        
    Returns:
        {"frequency":    , "warning":           None}
    """
    wave = analyze_code(code, "check")
    if wave is None:
        return {"frequency": 0, "warning": "     "}
    
    warning = None
    if wave.frequency > threshold:
        warning = f"             ({wave.frequency:.1f} > {threshold}).        ."
    
    return {
        "frequency": wave.frequency,
        "amplitude": wave.amplitude,
        "dimension": wave.dimension.name if wave.dimension else "UNKNOWN",
        "warning": warning
    }


# ============================================================
#       
# ============================================================

def get_system_status() -> dict:
    """                ."""
    return {
        "wave_coding_system": WAVE_CODING_AVAILABLE,
        "quaternion_dna": QUATERNION_DNA_AVAILABLE,
        "wave_coder": WAVE_CODER_AVAILABLE,
        "all_systems_ready": all([
            WAVE_CODING_AVAILABLE,
            QUATERNION_DNA_AVAILABLE,
            WAVE_CODER_AVAILABLE
        ])
    }


# ============================================================
# Export
# ============================================================

__all__ = [
    #      
    "analyze_code",
    "detect_resonance",
    "compress_to_dna",
    "decompress_from_dna",
    "transmute_codebase",
    "check_complexity",
    
    #       (NEW)
    "scan_quality",
    "WaveQualityGuard",
    
    #   
    "get_system_status",
    
    #   
    "CodeWave",
    "CodeDimension",
    "CodePhase",
    "QuaternionWaveDNA",
]


#            
try:
    from Core.1_Body.L5_Mental.Reasoning_Core.Physics_Waves.Wave.quality_guard import WaveQualityGuard, QualityReport
    
    def scan_quality(directory: str) -> "QualityReport":
        """
                  
        
        Args:
            directory:         
            
        Returns:
            QualityReport   
        """
        guard = WaveQualityGuard()
        return guard.scan_directory(directory)
    
    QUALITY_GUARD_AVAILABLE = True
except ImportError:
    QUALITY_GUARD_AVAILABLE = False
    
    def scan_quality(directory: str):
        logger.error("QualityGuard not available")
        return None


if __name__ == "__main__":
    print("=" * 60)
    print("  WAVE LANGUAGE UNIFIED API")
    print("=" * 60)
    
    status = get_system_status()
    print(f"\n  System Status:")
    for key, value in status.items():
        icon = " " if value else " "
        print(f"   {icon} {key}: {value}")
    
    if status["all_systems_ready"]:
        print("\n  Quick Demo:")
        
        #      
        wave = analyze_code("def add(a, b): return a + b", "demo.py")
        if wave:
            print(f"        : freq={wave.frequency}, dim={wave.dimension.name}")
        
        #       
        result = check_complexity("def simple(): pass")
        print(f"         : {result}")
        
        # DNA   
        dna = compress_to_dna("Hello, Wave!")
        if dna:
            restored = decompress_from_dna(dna)
            print(f"   DNA   /  : 'Hello, Wave!'   '{restored}'")
    
    print("\n  API ready!")
