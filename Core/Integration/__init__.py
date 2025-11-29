"""
__init__.py for Core.Integration module
Central hub for all system integration and communication.
"""

from .experience_digester import ExperienceDigester
from .integration_bridge import (
    IntegrationBridge,
    IntegrationEvent,
    EventType,
    ResonanceData,
    ConceptData,
    RelationshipData,
    ResonanceAdapter,
    HippocampusAdapter
)
from .communication_hub import (
    CommunicationHub,
    ModuleInterface,
    Signal,
    SignalType,
    create_signal
)
from .zelnaga_protocol import (
    ZelnagaProtocol,
    WaveUnifier,
    AlternativeCodeTranslator,
    WaveCodeGenerator,
    WillType,
    WillWave,
    WaveIntent,
    WaveCode,
    CodePatternType,
    CodePattern
)

__all__ = [
    # Experience Processing
    'ExperienceDigester',
    
    # Integration Bridge
    'IntegrationBridge',
    'IntegrationEvent',
    'EventType',
    'ResonanceData',
    'ConceptData',
    'RelationshipData',
    'ResonanceAdapter',
    'HippocampusAdapter',
    
    # Communication Hub
    'CommunicationHub',
    'ModuleInterface',
    'Signal',
    'SignalType',
    'create_signal',
    
    # Zelnaga Protocol
    'ZelnagaProtocol',
    'WaveUnifier',
    'AlternativeCodeTranslator', 
    'WaveCodeGenerator',
    'WillType',
    'WillWave',
    'WaveIntent',
    'WaveCode',
    'CodePatternType',
    'CodePattern'
]
