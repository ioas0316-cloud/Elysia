"""
Fractal Logging System

프랙탈 동형성 원칙: 모든 시스템이 같은 구조를 따른다.
Memory ≅ Log ≅ Document ≅ Context
"""

from .fractal_log import (
    FractalLogSphere,
    LogEntry,
    FractalLogHandler,
    get_fractal_logger,
    configure_fractal_logging
)

__all__ = [
    'FractalLogSphere',
    'LogEntry', 
    'FractalLogHandler',
    'get_fractal_logger',
    'configure_fractal_logging'
]
