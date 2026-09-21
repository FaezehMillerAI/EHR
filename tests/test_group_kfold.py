"""
Unit tests for data loader and GroupKFold cross-validation partitioning.
"""

import unittest
from pathlib import Path
from archehr_pipeline.config import PipelineConfig
from archehr_pipeline.data_loader import parse_cases_from_xml, get_5fold_cv_splits, cases_to_sentence_dataframe

class TestDataLoaderAndGroupKFold(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = PipelineConfig()
        cls.dev_cases = parse_cases_from_xml(cls.config.data_root / "dev", with_key=True)
        cls.test_cases = parse_cases_from_xml(cls.config.data_root / "test", with_key=True)

    def test_dev_cases_counts_and_annotations(self):
        self.assertEqual(len(self.dev_cases), 20)
        for c in self.dev_cases:
            self.assertIsNotNone(c.labels)
            self.assertGreater(len(c.sentences), 0)
            self.assertGreater(len(c.essential_sentence_ids), 0)
            self.assertTrue(bool(c.clinician_answer))

    def test_test_cases_counts(self):
        self.assertEqual(len(self.test_cases), 100)
        for c in self.test_cases:
            self.assertGreater(len(c.sentences), 0)
            self.assertTrue(bool(c.clinician_answer))
            # Test key should not have sentence relevance labels
            self.assertIsNone(c.labels)

    def test_group_kfold_no_leakage(self):
        splits = get_5fold_cv_splits(self.dev_cases)
        self.assertEqual(len(splits), 5)
        
        seen_val_cases = set()
        for fold_idx, (train_cases, val_cases) in enumerate(splits):
            train_ids = {c.case_id for c in train_cases}
            val_ids = {c.case_id for c in val_cases}

            # Exactly 16 train cases and 4 validation cases per fold
            self.assertEqual(len(train_cases), 16)
            self.assertEqual(len(val_cases), 4)

            # Strictly 0 case overlap between train and val within the fold
            self.assertEqual(len(train_ids & val_ids), 0, f"Leakage detected in fold {fold_idx}!")

            # Check that each case lands in val exactly once overall
            self.assertTrue(val_ids.isdisjoint(seen_val_cases), f"Case repeated in val across folds!")
            seen_val_cases.update(val_ids)

        self.assertEqual(len(seen_val_cases), 20)

    def test_graded_dataframe_mapping(self):
        df_or_rows = cases_to_sentence_dataframe(self.dev_cases)
        self.assertGreater(len(df_or_rows), 400)
        # Check graded values
        if hasattr(df_or_rows, "unique"):
            unique_labels = set(df_or_rows["label_graded"].unique())
        else:
            unique_labels = {r["label_graded"] for r in df_or_rows}
        self.assertTrue(unique_labels.issubset({0, 1, 2}))
        self.assertIn(2, unique_labels)  # essential
        self.assertIn(1, unique_labels)  # supplementary
        self.assertIn(0, unique_labels)  # not-relevant

if __name__ == "__main__":
    unittest.main()
