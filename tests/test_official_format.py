"""
Unit test for official submission parsing, 75-word whole-sentence truncation,
premise concatenation, and strict/lenient factuality computation.
"""

import unittest
from archehr_pipeline.generator import truncate_to_whole_sentences, count_words
from archehr_pipeline.nli_verifier import build_concatenated_premise
from archehr_pipeline.official_eval import (
    parse_submission_cases,
    compute_factuality_scores,
    compute_text_relevance_metrics,
    compute_overall_leaderboard
)

class TestOfficialFormatAndScorer(unittest.TestCase):
    def test_whole_sentence_truncation_to_75_words(self):
        sentence_1 = "The patient was diagnosed with severe common bile duct stones and elevated liver enzymes. |2, 6|"
        sentence_2 = "An urgent endoscopic retrograde cholangiopancreatography was performed to place a common duct stent. |6, 7|"
        sentence_3 = "Following stent replacement, liver enzymes stabilized and the patient improved markedly. |8|"
        sentence_4 = "The attending physician recommended completion of oral antibiotic therapy for two weeks with outpatient follow-up. |9|"
        long_tail = "Furthermore, multiple other observations, unrelated physical findings, and incidental laboratory evaluations were comprehensively documented during the intensive care hospital admission that are superfluous to the immediate query. |10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20|"

        combined = f"{sentence_1}\n{sentence_2}\n{sentence_3}\n{sentence_4}\n{long_tail}"
        truncated = truncate_to_whole_sentences(combined, max_words=75)

        # Word count must strictly be <= 75
        self.assertLessEqual(count_words(truncated), 75)
        # Trailing long sentence should have been cleanly dropped, leaving whole sentences
        self.assertIn("Following stent replacement", truncated)
        self.assertNotIn("superfluous", truncated)
        # Pipe citation format preserved on retained sentences
        self.assertTrue(truncated.endswith("|9|"))

    def test_build_concatenated_premise(self):
        sentence_map = {
            "2": "Pancreatic stent was placed to allow drainage.",
            "6": "Patient returned for re-evaluation as LFTs increased.",
            "8": "Retrograde cholangiogram was negative for filling defects."
        }
        premise = build_concatenated_premise(["2", "6"], sentence_map)
        self.assertEqual(
            premise,
            "Pancreatic stent was placed to allow drainage. Patient returned for re-evaluation as LFTs increased."
        )

    def test_factuality_scores_strict_and_lenient(self):
        # Case 1: gold essential = {2, 6}, supplementary = {3}
        key_map = {
            "1": {"1": "not-relevant", "2": "essential", "3": "supplementary", "6": "essential"}
        }

        # Submission predicts citations {2, 3}
        raw_submission = [{
            "case_id": "1",
            "answer": "Stent was placed for duct drainage. |2|\nMinor incision noted. |3|"
        }]
        parsed = parse_submission_cases(raw_submission)
        self.assertEqual(parsed[0]["citations"], {"2", "3"})

        scores = compute_factuality_scores(parsed, key_map)

        # Under Strict: essential is {2, 6}. Prediction is {2, 3}.
        # TP = {2} (len 1), FP = {3} (len 1), FN = {6} (len 1)
        # Precision = 1/2 = 0.5, Recall = 1/2 = 0.5, F1 = 0.5
        strict_micro = scores["strict"]["micro"]
        self.assertAlmostEqual(strict_micro["precision"], 0.5)
        self.assertAlmostEqual(strict_micro["recall"], 0.5)
        self.assertAlmostEqual(strict_micro["f1"], 0.5)

        # Under Lenient: allowed is {2, 3, 6}. Prediction is {2, 3}.
        # TP = {2, 3} (len 2), FP = {} (len 0), FN = {6} (len 1)
        # Precision = 2/2 = 1.0, Recall = 2/3 = 0.6667, F1 = 0.8
        lenient_micro = scores["lenient"]["micro"]
        self.assertAlmostEqual(lenient_micro["precision"], 1.0)
        self.assertAlmostEqual(lenient_micro["recall"], 2.0 / 3.0, places=3)
        self.assertAlmostEqual(lenient_micro["f1"], 0.8, places=3)

    def test_text_relevance_and_leaderboard(self):
        preds = ["Patient underwent ERCP to place a stent."]
        refs = ["Patient underwent ERCP for common bile duct stent."]
        rel = compute_text_relevance_metrics(preds, refs)
        self.assertIn("rougeLsum", rel)
        self.assertIn("bleu", rel)
        self.assertGreater(rel["rougeLsum"], 0.5)

        leaderboard = compute_overall_leaderboard(None, rel)
        self.assertIn("overall_relevance_score", leaderboard)

if __name__ == "__main__":
    unittest.main()
