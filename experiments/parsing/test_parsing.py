"""Behavioral checks for dependency matching, negation and graded-code isolation."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from factcheck import DependencyRecallThresholdFactChecker


class ParsingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.checker = DependencyRecallThresholdFactChecker()

    @staticmethod
    def passages(text, title=''):
        return [{'title': title, 'text': text}]

    def test_identical_supported_claim(self):
        self.assertEqual(self.checker.predict('Alice founded Nova.', self.passages('Alice founded Nova.')), 'S')

    def test_subject_object_reversal_is_not_supported(self):
        self.assertEqual(self.checker.predict('Alice founded Nova.', self.passages('Nova founded Alice.')), 'NS')

    def test_explicit_negation_is_not_positive_support(self):
        self.assertEqual(self.checker.score('Alice founded Nova.', self.passages('Alice did not found Nova.')), 0.0)

    def test_negative_claim_is_not_supported_by_positive_evidence(self):
        self.assertEqual(self.checker.score('Alice did not found Nova.', self.passages('Alice founded Nova.')), 0.0)

    def test_matching_negative_claim(self):
        self.assertEqual(self.checker.predict('Alice did not found Nova.', self.passages('Alice did not found Nova.')), 'S')

    def test_entity_name_and_pronoun_agree(self):
        self.assertEqual(self.checker.predict('Alice founded Nova.', self.passages('She founded Nova.', 'Alice')), 'S')

    def test_best_sentence_can_be_in_later_passage(self):
        passages = self.passages('Bob paints portraits.') + self.passages('Alice founded Nova.')
        self.assertEqual(self.checker.predict('Alice founded Nova.', passages), 'S')

    def test_empty_input_and_unrelated_evidence(self):
        self.assertEqual(self.checker.predict('', self.passages('Alice founded Nova.')), 'NS')
        self.assertEqual(self.checker.predict('Alice founded Nova.', []), 'NS')
        self.assertEqual(self.checker.predict('Alice founded Nova.', self.passages('Bob paints portraits.')), 'NS')

    def test_single_token_fallback(self):
        self.assertEqual(self.checker.predict('Alice', self.passages('Alice')), 'S')
        self.assertEqual(self.checker.predict('Alice', self.passages('Bob')), 'NS')

    def test_cached_prediction_is_stable(self):
        p = self.passages('Alice founded Nova.')
        first = self.checker.score('Alice founded Nova.', p)
        count = len(self.checker._cache)
        self.assertEqual(first, self.checker.score('Alice founded Nova.', p))
        self.assertEqual(count, len(self.checker._cache))

    def test_dependency_relations_have_lemmas_and_roles(self):
        triples = self.checker.get_dependencies('Alice founded Nova.')
        self.assertIn(('found', 'nsubj', 'alice'), triples)
        self.assertIn(('found', 'dobj', 'nova'), triples)

    def test_invalid_threshold_is_rejected_before_model_load(self):
        for threshold in [0.0, -0.1, 1.1, float('nan')]:
            with self.assertRaises(ValueError):
                DependencyRecallThresholdFactChecker(threshold)

    def test_invalid_dependency_weight_is_rejected(self):
        for weight in [0.0, -0.1, 1.1, float('nan')]:
            with self.assertRaises(ValueError):
                DependencyRecallThresholdFactChecker(dependency_weight=weight)

    def test_lexical_backoff_can_join_compatible_sentences(self):
        fact = 'Alice is an American actress.'
        passages = self.passages('Alice is American. She is an actress.', 'Alice')
        dependency, lexical = self.checker.score_components(fact, passages)
        self.assertLess(dependency, 0.9)
        self.assertEqual(lexical, 1.0)
        self.assertEqual(self.checker.predict(fact, passages), 'S')

    def test_reversal_is_rejected_at_the_deployed_hybrid_threshold(self):
        passages = self.passages('Nova founded Alice.')
        self.assertEqual(self.checker.score_components('Alice founded Nova.', passages), (0.0, 0.0))
        self.assertEqual(self.checker.predict('Alice founded Nova.', passages), 'NS')

    def test_compatible_sentence_survives_a_contradictory_distractor(self):
        passages = self.passages('Alice did not found Nova. Alice founded Nova.')
        self.assertEqual(self.checker.predict('Alice founded Nova.', passages), 'S')

    def test_missing_parser_has_actionable_message(self):
        with patch('factcheck.spacy.load', side_effect=OSError('missing')):
            with self.assertRaisesRegex(RuntimeError, 'python -m spacy download en_core_web_sm'):
                DependencyRecallThresholdFactChecker()


if __name__ == '__main__':
    unittest.main(verbosity=2)
