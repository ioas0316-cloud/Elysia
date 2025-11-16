from core.concept_os.kernel_v2 import ConceptKernelV2, ConceptMessage

class ChronosConceptBridge:
    def __init__(self, kernel: ConceptKernelV2):
        self.kernel = kernel
    def on_rewind(self, steps:int):
        self.kernel.post(ConceptMessage(concept_id="time.rewind", vector=[steps], tags=["chronos"], priority=0.8))
