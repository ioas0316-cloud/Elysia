"""
Kernel Validation Module

검증 관련 함수들
"""

import sys
import os
import logging
import time
import json

def _check_values(self, input_text: str, response: str) -> None:
    """Lightweight value alignment guard; logs if core values are missing/drifting."""
    core_words = list(self.core_values.keys())
    text = (input_text + ' ' + response).lower()
    if not any((w in text for w in core_words)):
        logger.warning('[VALUES] Core values not referenced in recent exchange.')
