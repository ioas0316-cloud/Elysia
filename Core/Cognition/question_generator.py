"""
Question Generator for Elysia's Truth Seeker System

This module takes a notable hypothesis (e.g., a strong co-occurrence of two concepts)
and formulates a natural language question in Korean to ask the user for verification.
"""
from typing import Dict, Optional, Tuple

class QuestionGenerator:
    def __init__(self):
        # Relationship templates
        # Using format placeholders: {head_subject}, {tail_object}, etc.
        # These will be populated after processing josa (particles)
        self.templates = {
            "is_a": "  , {head_subject} {tail}         ?",
            "causes": "  , {head_subject} {tail_object}             ?",
            "enables": "  , {head_subject}     {tail_subject}       ?",
            "prevents": "  , {head_subject} {tail_object}          ?",
            "creates": "  , {head_subject} {tail_object}       ?",
            "is_composed_of": "  , {head_subject} {tail_instrument}         ?",
            "related_to": "  ,                               .    {head_with_wa} {tail_subject}                  ?"
        }

    def _get_batchim_code(self, char: str) -> int:
        """
        Returns the batchim code for the last character.
        Returns 0 if no batchim or not Hangul.
        """
        if not char:
            return 0

        code = ord(char[-1])
        if 0xAC00 <= code <= 0xD7A3:
            return (code - 0xAC00) % 28
        return 0

    def _has_batchim(self, char: str) -> bool:
        """
        Checks if the last character has a final consonant (batchim).
        """
        return self._get_batchim_code(char) != 0

    def _attach_josa(self, word: str, josa_pair: Tuple[str, str]) -> str:
        """
        Attaches the correct particle (josa) based on the final sound of the word.

        Args:
            word: The word to attach the particle to.
            josa_pair: A tuple of (josa_with_batchim, josa_without_batchim).
                       e.g., (' ', ' '), (' ', ' '), (' ', ' '), (' ', ' '), ('  ', ' ')

        Returns:
            The word combined with the correct particle.
        """
        if not word:
            return ""

        batchim_code = self._get_batchim_code(word)
        has_batchim = batchim_code != 0

        # Special handling for ( ) 
        if josa_pair == ('  ', ' '):
            # If batchim exists and is NOT ' ' (code 8), use '  '
            # ' ' batchim behaves like a vowel for this particle
            if has_batchim and batchim_code != 8:
                return f"{word}{josa_pair[0]}"
            else:
                return f"{word}{josa_pair[1]}"

        if has_batchim:
            return f"{word}{josa_pair[0]}"
        else:
            return f"{word}{josa_pair[1]}"

    def generate_question_from_hypothesis(self, hypothesis: Dict[str, any]) -> Optional[str]:
        """
        Generates a natural language question from a hypothesis dictionary.

        Args:
            hypothesis: A dictionary containing at least 'head', 'tail', and optionally 'relation'.

        Returns:
            A formatted question string, or None if the hypothesis is invalid.
        """
        head = hypothesis.get('head')
        tail = hypothesis.get('tail')
        relation = hypothesis.get('relation', 'related_to')

        if not head or not tail:
            return None

        # Pre-calculate common josa combinations
        head_subject = self._attach_josa(head, (' ', ' '))
        head_object = self._attach_josa(head, (' ', ' '))
        head_with_wa = self._attach_josa(head, (' ', ' '))

        tail_subject = self._attach_josa(tail, (' ', ' ')) # Changed default to i/ga for subjects in templates
        tail_object = self._attach_josa(tail, (' ', ' '))
        tail_with_wa = self._attach_josa(tail, (' ', ' '))
        tail_instrument = self._attach_josa(tail, ('  ', ' '))

        # Select template
        template = self.templates.get(relation, self.templates['related_to'])

        # Format the question
        try:
            # Note: tail_subject above uses i/ga, but for 'related_to' we might want eun/neun if it's the topic?'
            # actually related_to template uses {tail_subject}, which currently maps to i/ga above.
            # Original related_to: "   {head}( )  {tail}( ) ..." -> used eun/neun.
            # My current 'tail_subject' uses i/ga.
            # Let's fix this contextually.

            # For 'related_to', we specifically want ' / ' for the tail
            tail_topic = self._attach_josa(tail, (' ', ' '))

            # Context-specific overrides
            if relation == 'related_to':
                 question = template.format(
                    head_with_wa=head_with_wa,
                    tail_subject=tail_topic # Override to use eun/neun
                )
            else:
                question = template.format(
                    head=head,
                    tail=tail,
                    head_subject=head_subject,
                    head_object=head_object,
                    head_with_wa=head_with_wa,
                    tail_subject=tail_subject,
                    tail_object=tail_object,
                    tail_with_wa=tail_with_wa,
                    tail_instrument=tail_instrument
                )
        except KeyError:
            # Fallback
            tail_topic = self._attach_josa(tail, (' ', ' '))
            question = self.templates['related_to'].format(
                head_with_wa=head_with_wa,
                tail_subject=tail_topic
            )

        return question

    def generate_wisdom_seeking_question(self, hypothesis: Dict[str, any]) -> Optional[str]:
        """
        Generates a question that seeks wisdom or opinion, especially for 'forms_new_concept' hypotheses.
        """
        head = hypothesis.get('head')
        tail = hypothesis.get('tail')
        relation = hypothesis.get('relation')
        new_concept = hypothesis.get('new_concept_id')

        if not all([head, tail, relation, new_concept]):
            return None

        if relation == 'forms_new_concept':
            head_wa = self._attach_josa(head, (' ', ' '))
            tail_i = self._attach_josa(tail, (' ', ' '))

            question = (f"   ,            {head_wa} {tail_i}      "
                        f"'{new_concept}'                       . "
                        f"                  ?                          ?")
            return question

        # Fallback for other relations if needed, or just use the standard generator
        return self.generate_question_from_hypothesis(hypothesis)

    def generate_correction_proposal_question(self, hypothesis: Dict[str, any]) -> Optional[str]:
        """
        Generates a question for a correction proposal, prioritizing the text in the hypothesis.
        """
        # The hypothesis generated by the Guardian should contain the question text.
        if 'text' in hypothesis and hypothesis['text']:
            return hypothesis['text']

        # Fallback template if the text is missing for some reason
        head = hypothesis.get('head')
        tail = hypothesis.get('tail')
        if not all([head, tail]):
            return None

        head_wa = self._attach_josa(head, (' ', ' '))

        question = (f"   , {head_wa} '{tail}'                          . "
                    f"                         ,          ?")
        return question

if __name__ == '__main__':
    # Example usage for testing
    gen = QuestionGenerator()

    hypo1 = {"head": "  ", "tail": "  ", "confidence": 0.8}
    question1 = gen.generate_question_from_hypothesis(hypo1)
    print(f"Hypothesis: {hypo1}")
    print(f"Generated Question: {question1}")
    # Expected:   ,                               .                             ?

    hypo2 = {"head": "  ", "tail": "  ", "confidence": 1.0}
    question2 = gen.generate_question_from_hypothesis(hypo2)
    print(f"\nHypothesis: {hypo2}")
    print(f"Generated Question: {question2}")
    # Expected:   ,                               .                             ?
