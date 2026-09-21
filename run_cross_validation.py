"""
Run 5-fold cross-validation (out-of-fold) on ArchEHR-QA development set.
Evaluates:
  1. Perspective Ablation (P1 - P4) on sentence ranking
  2. Ablation Ladder (M0, M1, Control A, Control B, M2, M3)
  3. Pre-pruning sentence calibration (AUROC, ECE, Brier score with bootstrap CI)
  4. Out-of-fold Factuality (Strict/Lenient F1) and Relevance (BLEU, ROUGE-Lsum)
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    _HAS_NUMPY = False
    class _DummyNP:
        @staticmethod
        def mean(vals):
            vals_list = list(vals)
            return sum(vals_list) / len(vals_list) if vals_list else 0.0
    np = _DummyNP()

from archehr_pipeline.config import PipelineConfig
from archehr_pipeline.data_loader import parse_cases_from_xml, get_5fold_cv_splits
from archehr_pipeline.sentence_ranker import MultiPerspectiveSentenceRanker
from archehr_pipeline.generator import ClinicalAnswerGenerator
from archehr_pipeline.sc_cal import SampleConsistencyCalibrator, compute_calibration_metrics
from archehr_pipeline.nli_verifier import NLIClaimVerifier
from archehr_pipeline.official_eval import (
    parse_submission_cases,
    compute_factuality_scores,
    compute_text_relevance_metrics,
    compute_overall_leaderboard
)


def run_perspective_ablation(cases: List[Any], config: PipelineConfig) -> Dict[str, Any]:
    """
    Evaluate ranking performance across perspectives P1-P4 on dev cases.
      P1: Patient Question only
      P2: Clinician Question only
      P3: Patient Question + Clinician Question
      P4: Full Multi-Perspective (Clinician Q + Note Sentence + Narrative[:300 chars])
    """
    print("\n--- Running Perspective Ablation (P1 - P4) ---")
    ranker = MultiPerspectiveSentenceRanker(config)
    results = {}

    for mode in ["P1", "P2", "P3", "P4"]:
        recalls_at_k = []
        precisions_at_k = []
        mrrs = []

        for c in cases:
            gold_ess = set(c.essential_sentence_ids)
            if not gold_ess:
                continue

            scored_sents = ranker.score_sentences(c, perspective_mode=mode)
            top_k_ids = [s["id"] for s in scored_sents[:config.k_context]]

            # Recall@K
            hits = len(set(top_k_ids) & gold_ess)
            recalls_at_k.append(hits / len(gold_ess))
            # Precision@K
            precisions_at_k.append(hits / max(1, len(top_k_ids)))

            # MRR (first essential hit)
            first_rank = None
            for r_idx, s in enumerate(scored_sents):
                if s["id"] in gold_ess:
                    first_rank = r_idx + 1
                    break
            mrrs.append(1.0 / first_rank if first_rank else 0.0)

        results[mode] = {
            f"Recall@{config.k_context}": float(np.mean(recalls_at_k)),
            f"Precision@{config.k_context}": float(np.mean(precisions_at_k)),
            "MRR": float(np.mean(mrrs))
        }
        print(f"  {mode}: Recall@{config.k_context}={results[mode][f'Recall@{config.k_context}']:.4f}, "
              f"Precision@{config.k_context}={results[mode][f'Precision@{config.k_context}']:.4f}, "
              f"MRR={results[mode]['MRR']:.4f}")

    return results


def run_5fold_cross_validation(
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    config: Optional[PipelineConfig] = None
) -> Dict[str, Any]:
    """
    Execute out-of-fold cross-validation across the 20 dev cases.
    """
    config = config or PipelineConfig()
    dev_cases = parse_cases_from_xml(config.data_root / "dev", with_key=True)
    splits = get_5fold_cv_splits(dev_cases)

    # Initialize modules
    ranker = MultiPerspectiveSentenceRanker(config)
    generator = ClinicalAnswerGenerator(model_name, config)
    calibrator = SampleConsistencyCalibrator(config)
    verifier = NLIClaimVerifier(config)

    # Storage for out-of-fold predictions per ablation variant
    oof_variants: Dict[str, List[Dict[str, Any]]] = {
        "M0": [],        # Unranked full context greedy
        "M1": [],        # Top-K reranked context greedy
        "Control_A": [], # Single stochastic rollout (sample 0)
        "Control_B": [], # Random stochastic rollout (sample rand)
        "M2": [],        # Medoid consensus (pre-pruning)
        "M3": []         # Full system (Medoid + NLI-SHP pruning)
    }

    # Calibration tracking (sentence-level on pre-pruning medoid)
    all_sentence_scores = []
    all_sentence_labels = []

    print(f"\nStarting 5-Fold Grouped Cross-Validation with {model_name}...")
    for fold_idx, (train_cases, val_cases) in enumerate(splits):
        print(f"\n=== Fold {fold_idx + 1}/5 (Train: {len(train_cases)} cases, Val: {len(val_cases)} cases) ===")

        # In full training mode, fine-tune ranker on train_cases:
        # ranker.train_on_cases(train_cases, epochs=config.ranker_epochs)

        for case in val_cases:
            # 1. Reranking
            top_k_sentences = ranker.select_top_k_context(case, k=config.k_context, perspective_mode="P4")

            # 2. Generation Variants
            # M0: Full raw context greedy
            ans_m0 = generator.generate_greedy(case, case.sentences)
            oof_variants["M0"].append({"case_id": case.case_id, "answer": ans_m0})

            # M1: Top-K reranked context greedy
            ans_m1 = generator.generate_greedy(case, top_k_sentences)
            oof_variants["M1"].append({"case_id": case.case_id, "answer": ans_m1})

            # Rollouts: Generate R=10 stochastic completions (shared for Control A, B, and M2)
            rollouts = generator.generate_rollouts(case, top_k_sentences, r_rollouts=config.r_rollouts)

            # Control A: First sample (T=0.7)
            ans_ctrl_a = rollouts[0] if rollouts else ans_m1
            oof_variants["Control_A"].append({"case_id": case.case_id, "answer": ans_ctrl_a})

            # Control B: Random sample
            import random
            rand_idx = random.Random(config.seed + int(case.case_id)).randint(0, len(rollouts) - 1) if rollouts else 0
            ans_ctrl_b = rollouts[rand_idx] if rollouts else ans_m1
            oof_variants["Control_B"].append({"case_id": case.case_id, "answer": ans_ctrl_b})

            # M2: Medoid selection & sentence consistency
            med_idx, med_ans, consensus_score, _ = calibrator.select_medoid(rollouts)
            oof_variants["M2"].append({"case_id": case.case_id, "answer": med_ans})

            # Sentence consistency c(s_k) on pre-pruning medoid
            med_sents = calibrator.compute_sentence_consistency(med_ans, rollouts, med_idx)

            # Ground-truth evaluation on pre-pruning medoid sentences
            labeled_sents = verifier.evaluate_ground_truth_sentence_labels(med_sents, case, strict=True)
            for ls in labeled_sents:
                all_sentence_scores.append(ls["raw_consistency"])
                all_sentence_labels.append(ls["gold_y"])

            # M3: Full System (Medoid + NLI-SHP verification and pruning)
            ans_m3, _ = verifier.verify_and_prune(med_sents, case, top_k_sentences, theta_ungrounded=config.theta_ungrounded)
            oof_variants["M3"].append({"case_id": case.case_id, "answer": ans_m3})

    # Fit Platt scaling on out-of-fold calibration samples
    calibrator.fit_platt_scaling(all_sentence_scores, all_sentence_labels)
    cal_metrics = compute_calibration_metrics(all_sentence_scores, all_sentence_labels, num_bins=10, n_bootstrap=1000)

    # Key map for official evaluation
    key_map = {c.case_id: c.labels for c in dev_cases if c.labels}

    # Evaluate official metrics for each ablation variant
    evaluation_summary = {}
    print("\n--- Out-of-Fold Evaluation Results ---")
    for variant, raw_preds in oof_variants.items():
        parsed = parse_submission_cases(raw_preds, max_answer_words=config.max_answer_words)
        fact_scores = compute_factuality_scores(parsed, key_map)

        preds_text = [p["answer"] for p in parsed]
        refs_text = [c.clinician_answer for c in dev_cases]
        rel_scores = compute_text_relevance_metrics(preds_text, refs_text)

        leaderboard = compute_overall_leaderboard(fact_scores, rel_scores)
        evaluation_summary[variant] = leaderboard

        print(f"Variant [{variant}]: "
              f"Strict Micro F1={leaderboard['strict_micro_f1']:.2f}, "
              f"Lenient Micro F1={leaderboard['lenient_micro_f1']:.2f}, "
              f"ROUGE-Lsum={leaderboard['rougeLsum']:.2f}, "
              f"BLEU={leaderboard['bleu']:.2f}, "
              f"Overall Score={leaderboard['overall_score']:.2f}")

    print("\n--- SC-Cal Calibration Metrics (Pre-Pruning Medoid) ---")
    print(f"  AUROC: {cal_metrics['auroc']:.4f} (95% CI: {cal_metrics['auroc_ci'][0]:.4f} - {cal_metrics['auroc_ci'][1]:.4f})")
    print(f"  ECE:   {cal_metrics['ece']:.4f} (95% CI: {cal_metrics['ece_ci'][0]:.4f} - {cal_metrics['ece_ci'][1]:.4f})")
    print(f"  Brier: {cal_metrics['brier']:.4f} (95% CI: {cal_metrics['brier_ci'][0]:.4f} - {cal_metrics['brier_ci'][1]:.4f})")

    # Perspective ablation
    perspective_results = run_perspective_ablation(dev_cases, config)

    output_payload = {
        "model_name": model_name,
        "calibration": cal_metrics,
        "ablation_summary": evaluation_summary,
        "perspective_ablation": perspective_results
    }

    out_file = config.output_dir / "dev_oof_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)
    print(f"\nSaved out-of-fold results to {out_file}")

    return output_payload


if __name__ == "__main__":
    run_5fold_cross_validation()
