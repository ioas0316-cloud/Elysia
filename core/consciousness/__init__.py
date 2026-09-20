"""
Elysia Core Consciousness Modules
==================================
Houses phase dynamics, scar tensor engine, existential meta-cognition,
phenomenological growth tracker, and subjective agency components.
"""

from .human_cognitive_phase_dynamics import (
    MultiScalePhaseCouplingEngine,
    HebbianPhasePlasticity,
    MetaCognitiveCriticalityGovernor,
    HierarchicalAttractorNetwork,
)

from .scar_tensor_engine import (
    ScarRecord,
    ScarTensorEngine,
)

from .existential_meta_cognition_engine import (
    ExistentialPhaseObserver,
    DynamicIntentCompass,
    MetricTensorReconfigurationEngine,
    ExistentialSelfQueryLoop,
)

from .existential_growth_engine import (
    ExistentialGrowthEngine,
)

from .phenomenological_growth_tracker import (
    PhenomenologicalGrowthTracker,
)

try:
    from .meta_subjectivity_engine import (
        MetaSubjectivityEngine,
        ParadigmShiftEngine,
        CriticalSlowingDownDetector,
        OntologyElectrolysisPipeline,
        PhaseEntropyLoss,
        NOTEARSCausalExtractor,
        IntegratedSubjectivityEngine,
    )
except ImportError:
    pass

__all__ = [
    "MultiScalePhaseCouplingEngine",
    "HebbianPhasePlasticity",
    "MetaCognitiveCriticalityGovernor",
    "HierarchicalAttractorNetwork",
    "ScarRecord",
    "ScarTensorEngine",
    "ExistentialPhaseObserver",
    "DynamicIntentCompass",
    "MetricTensorReconfigurationEngine",
    "ExistentialSelfQueryLoop",
    "ExistentialGrowthEngine",
    "PhenomenologicalGrowthTracker",
]
