"""
Unit test for SC-Cal sentence-level consistency scoring and Platt calibration.
"""

import unittest
from archehr_pipeline.sc_cal import (
    SampleConsistencyCalibrator,
    extract_sentences_with_citations,
    compute_calibration_metrics
)
from archehr_pipeline.generator import strip_citations

class TestSampleConsistencyAndCalibration(unittest.TestCase):
    def setUp(self):
        self.calibrator = SampleConsistencyCalibrator()

    def test_strip_citations(self):
        raw = "The patient was prescribed udiliv |2, 6| and discharged home |8|."
        cleaned = strip_citations(raw)
        self.assertEqual(cleaned, "The patient was prescribed udiliv and discharged home.")

    def test_extract_sentences_with_citations(self):
        answer = (
            "Patient was diagnosed with common bile duct stones. |2|\n"
            "An ERCP was successfully performed to place a biliary stent. |6, 7|\n"
            "Post-procedure antibiotics were administered. |8|"
        )
        parsed = extract_sentences_with_citations(answer)
        self.assertEqual(len(parsed), 3)
        self.assertEqual(parsed[0]["citations"], ["2"])
        self.assertEqual(parsed[1]["citations"], ["6", "7"])
        self.assertEqual(parsed[2]["citations"], ["8"])

    def test_medoid_selection_and_sentence_consistency(self):
        rollouts = [
            "ERCP was recommended to remove common bile duct stones and place a stent. |2, 6|",
            "A repeat ERCP was necessary due to recurrent biliary duct obstruction. |6, 7|",
            "ERCP was recommended to remove common bile duct stones and place a stent. |2, 6|",
            "Emergency surgery was performed for acute bile duct obstruction. |3|",
        ]
        med_idx, med_ans, consensus, sim_mat = self.calibrator.select_medoid(rollouts)
        self.assertIn(med_idx, [0, 2])  # 0 and 2 are identical consensus answers
        self.assertGreater(consensus, 0.0)

        # Compute sentence consistency c(s_k)
        scored_sents = self.calibrator.compute_sentence_consistency(med_ans, rollouts, med_idx)
        self.assertEqual(len(scored_sents), 1)
        self.assertIn("raw_consistency", scored_sents[0])
        self.assertIn("calibrated_prob", scored_sents[0])
        self.assertGreater(scored_sents[0]["raw_consistency"], 0.0)

    def test_platt_scaling_and_calibration_metrics(self):
        scores = [0.95, 0.90, 0.88, 0.85, 0.40, 0.35, 0.30, 0.25]
        labels = [1, 1, 1, 1, 0, 0, 0, 0]
        self.calibrator.fit_platt_scaling(scores, labels)

        prob_high = self.calibrator.apply_platt_scaling(0.92)
        prob_low = self.calibrator.apply_platt_scaling(0.30)
        self.assertGreater(prob_high, prob_low)

        metrics = compute_calibration_metrics(scores, labels, num_bins=5, n_bootstrap=100)
        self.assertGreaterEqual(metrics["auroc"], 0.90)
        self.assertIn("ece", metrics)
        self.assertIn("brier", metrics)
        self.assertIn("auroc_ci", metrics)

if __name__ == "__main__":
    unittest.main()
