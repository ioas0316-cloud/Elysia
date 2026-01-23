    def _create_concept_seed_from_definition(self, concept_def: ConceptDefinition) -> ConceptNode:
        """        Seed   """
        # Quaternion:        (4D)
        orientation = self._concept_definition_to_quaternion(concept_def)
        
        # Frequency
        freq = self._concept_to_frequency(concept_def.name)
        
        # ConceptNode   
        seed = ConceptNode(
            name=concept_def.name,
            frequency=freq,
            orientation=orientation,
            energy=1.0,
            depth=0
        )
        
        # metadata        (JSON   !)
        if not hasattr(seed, 'metadata'):
            seed.metadata = {}
        
        seed.metadata = {
            'description': concept_def.description,
            'properties': concept_def.properties,
            'type': concept_def.type,
            'context': concept_def.context
        }
        
        logger.info(f"  Seed: {concept_def.name} = {concept_def.description[:40]}...")
        
        return seed
    
    def _concept_definition_to_quaternion(self, concept_def: ConceptDefinition) -> Quaternion:
        """       Quaternion      (    )"""
        # w:     (              )
        w = 0.8 if concept_def.description else 0.3
        
        # x:      
        x = 0.0
        if concept_def.type == 'emotion':
            x = 0.9
            if 'positive' in concept_def.properties.get('valence', ''):
                x += 0.1
        
        # y:        
        y = 0.0
        if concept_def.type in ['action', 'object']:
            y = 0.7
        
        # z:      
        z = 0.0
        if 'good' in concept_def.description.lower() or 'bad' in concept_def.description.lower():
            z = 0.6
        
        return Quaternion(w, x, y, z).normalize()
    
    def _store_relationship(self, rel: Relationship):
        """    ResonanceField    """
        #              bloom
        source_seed = self.hippocampus.load_fractal_concept(rel.source)
        target_seed = self.hippocampus.load_fractal_concept(rel.target)
        
        # Bloom (ResonanceField    )
        if source_seed:
            self.resonance_field.inject_fractal_concept(source_seed, active=False)
        if target_seed:
            self.resonance_field.inject_fractal_concept(target_seed, active=False)
        
        #      
        if source_seed and target_seed:
            if rel.source in self.resonance_field.nodes and rel.target in self.resonance_field.nodes:
                self.resonance_field._connect(rel.source, rel.target)
                logger.info(f"  {rel.source} --{rel.type}--> {rel.target}")