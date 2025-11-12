from core.concept_os.kernel_v2 import ConceptKernelV2, ConceptMessage

class SpatialConceptBridge:
    def __init__(self, kernel: ConceptKernelV2):
        self.kernel = kernel
    def on_compress(self, region_id:str, factor:float):
        self.kernel.post(ConceptMessage(concept_id="space.compress", vector=[factor], tags=["aether", region_id], priority=0.7))
