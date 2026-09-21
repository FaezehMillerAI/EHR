"""
Standalone test benchmark runner for ArchEHR-QA Subtask 3.
Evaluates the 100 test cases with frozen hyperparameters tuned on dev:
  - Multi-Perspective Cross-Encoder (P4) ranking
  - 4-bit NF4 batched generation across 3 SLMs and 3 LLMs
  - MBR medoid consensus (SC-Cal) from R=10 rollouts
  - NLI-SHP verification and citation pruning
  - Strict <= 75 word whole-sentence truncation and pipe citation format
  - Relevance evaluation (ROUGE-1, ROUGE-2, ROUGE-Lsum, BLEU) against gold clinician answers
  - Paired bootstrap hypothesis testing (N=1,000 resamples) with Holm-Bonferroni correction
  - Official submission JSON generation
"""

import os
import sys
import json
import argparse
import random
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
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

        @staticmethod
        def percentile(vals, q):
            sorted_v = sorted(vals)
            if not sorted_v:
                return 0.0
            idx = int((q / 100.0) * (len(sorted_v) - 1))
            return sorted_v[idx]

        @staticmethod
        def sort(vals):
            return sorted(vals)

        @staticmethod
        def argsort(vals):
            return sorted(range(len(vals)), key=lambda i: vals[i])

        @staticmethod
        def array(vals):
            return list(vals)

    np = _DummyNP()

from archehr_pipeline.config import PipelineConfig
from archehr_pipeline.data_loader import parse_cases_from_xml, Case
from archehr_pipeline.sentence_ranker import MultiPerspectiveSentenceRanker
from archehr_pipeline.generator import ClinicalAnswerGenerator, truncate_to_whole_sentences
from archehr_pipeline.sc_cal import SampleConsistencyCalibrator
from archehr_pipeline.nli_verifier import NLIClaimVerifier
from archehr_pipeline.official_eval import (
    parse_submission_cases,
    compute_text_relevance_metrics,
    compute_overall_leaderboard
)

# Supported 3 SLMs + 3 LLMs
DEFAULT_MODELS = {
    "slms": [
        "Qwen/Qwen2.5-3B-Instruct",
        "meta-llama/Llama-3.2-3B-Instruct",
        "microsoft/Phi-3.5-mini-instruct"
    ],
    "llms": [
        "mistralai/Mistral-7B-Instruct-v0.3",
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "Qwen/Qwen2.5-7B-Instruct"
    ]
}


def paired_bootstrap_test(
    scores_a: List[float],
    scores_b: List[float],
    n_resamples: int = 1000,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Perform paired bootstrap test for difference in means (scores_a - scores_b).
    Returns mean difference, 95% bootstrap CI, and one-sided p-value for H1: a > b.
    """
    assert len(scores_a) == len(scores_b), "Score lists must have identical length"
    n = len(scores_a)
    if n == 0:
        return {"mean_diff": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "p_value": 1.0}

    obs_diff = float(np.mean([x - y for x, y in zip(scores_a, scores_b)]))

    if _HAS_NUMPY:
        a = np.array(scores_a)
        b = np.array(scores_b)
        rng = np.random.RandomState(seed)
        boot_diffs = []
        for _ in range(n_resamples):
            idx = rng.randint(0, n, size=n)
            boot_diffs.append(float(np.mean(a[idx] - b[idx])))
        boot_diffs = np.sort(boot_diffs)
        ci_lower = float(np.percentile(boot_diffs, 2.5))
        ci_upper = float(np.percentile(boot_diffs, 97.5))
        p_val = float(np.mean(boot_diffs <= 0.0))
    else:
        rng = random.Random(seed)
        boot_diffs = []
        diffs = [x - y for x, y in zip(scores_a, scores_b)]
        for _ in range(n_resamples):
            sample = rng.choices(diffs, k=n)
            boot_diffs.append(sum(sample) / n)
        boot_diffs.sort()
        ci_lower = boot_diffs[int(0.025 * len(boot_diffs))]
        ci_upper = boot_diffs[int(0.975 * len(boot_diffs))]
        p_val = sum(1 for d in boot_diffs if d <= 0.0) / len(boot_diffs)

    p_val = max(p_val, 1.0 / (n_resamples + 1))

    return {
        "mean_diff": obs_diff,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "p_value": p_val
    }


def holm_bonferroni_correction(p_values: List[float]) -> List[float]:
    """
    Apply Holm-Bonferroni step-down correction to a list of p-values.
    """
    m = len(p_values)
    if m == 0:
        return []

    # Sort indices by p-value
    sorted_indices = np.argsort(p_values)
    adjusted = [0.0] * m

    cum_max = 0.0
    for rank, orig_idx in enumerate(sorted_indices):
        multiplier = m - rank
        adj = min(1.0, multiplier * p_values[orig_idx])
        cum_max = max(cum_max, adj)
        adjusted[orig_idx] = cum_max

    return adjusted


def evaluate_test_model(
    model_name: str,
    test_cases: List[Case],
    config: PipelineConfig,
    save_rollouts: bool = False
) -> Dict[str, Any]:
    """
    Execute full pipeline and ablation variants on test cases for a single model.
    """
    print(f"\n=======================================================")
    print(f"Evaluating Model: {model_name}")
    print(f"Total Test Cases: {len(test_cases)}")
    print(f"=======================================================")

    ranker = MultiPerspectiveSentenceRanker(config)
    generator = ClinicalAnswerGenerator(model_name, config)
    calibrator = SampleConsistencyCalibrator(config)
    verifier = NLIClaimVerifier(config)

    # Predictions per ablation variant
    variants = {
        "M0": [],        # Unranked full context greedy
        "M1": [],        # Top-K reranked context greedy
        "Control_A": [], # First stochastic rollout (T=0.7)
        "Control_B": [], # Random stochastic rollout
        "M2": [],        # Medoid consensus (pre-pruning)
        "M3": []         # Full system (Medoid + NLI-SHP pruning)
    }

    all_rollouts = {}

    for idx, case in enumerate(test_cases):
        if (idx + 1) % 10 == 0 or idx == 0:
            print(f"  Processing Case {idx + 1}/{len(test_cases)} [ID: {case.case_id}]...")

        # 1. Multi-Perspective Ranking (P4)
        top_k = ranker.select_top_k_context(case, k=config.k_context, perspective_mode="P4")

        # 2. M0 (unranked full context) & M1 (reranked context)
        ans_m0 = generator.generate_greedy(case, case.sentences)
        ans_m1 = generator.generate_greedy(case, top_k)
        variants["M0"].append({"case_id": case.case_id, "answer": ans_m0})
        variants["M1"].append({"case_id": case.case_id, "answer": ans_m1})

        # 3. Stochastic Rollouts (R=10, T=0.7, p=0.9)
        rollouts = generator.generate_rollouts(case, top_k, r_rollouts=config.r_rollouts)
        if save_rollouts:
            all_rollouts[case.case_id] = rollouts

        # Control A: First sample
        ans_ctrl_a = rollouts[0] if rollouts else ans_m1
        variants["Control_A"].append({"case_id": case.case_id, "answer": ans_ctrl_a})

        # Control B: Random sample (seeded by case_id)
        rand_idx = random.Random(config.seed + int(case.case_id)).randint(0, len(rollouts) - 1) if rollouts else 0
        ans_ctrl_b = rollouts[rand_idx] if rollouts else ans_m1
        variants["Control_B"].append({"case_id": case.case_id, "answer": ans_ctrl_b})

        # M2: Medoid selection
        med_idx, med_ans, _, _ = calibrator.select_medoid(rollouts)
        variants["M2"].append({"case_id": case.case_id, "answer": med_ans})

        # Sentence consistency c(s_k)
        med_sents = calibrator.compute_sentence_consistency(med_ans, rollouts, med_idx)

        # M3: Full System (Medoid + NLI-SHP verification & citation pruning)
        ans_m3, _ = verifier.verify_and_prune(med_sents, case, top_k, theta_ungrounded=config.theta_ungrounded)
        variants["M3"].append({"case_id": case.case_id, "answer": ans_m3})

    # Evaluate metrics against clinician answers
    refs_text = [c.clinician_answer for c in test_cases]
    variant_results = {}
    variant_per_case_metrics = {}

    for var_name, preds in variants.items():
        parsed = parse_submission_cases(preds, max_answer_words=config.max_answer_words)
        preds_text = [p["answer"] for p in parsed]

        # Compute aggregate relevance
        rel_scores = compute_text_relevance_metrics(preds_text, refs_text)
        leaderboard = compute_overall_leaderboard(factuality_scores=None, relevance_scores=rel_scores)
        variant_results[var_name] = leaderboard

        # Compute per-case scores for paired bootstrap
        per_case_rouge = []
        per_case_bleu = []
        for p, r in zip(preds_text, refs_text):
            sc = compute_text_relevance_metrics([p], [r])
            per_case_rouge.append(sc["rougeLsum"])
            per_case_bleu.append(sc["bleu"])

        variant_per_case_metrics[var_name] = {
            "rougeLsum": per_case_rouge,
            "bleu": per_case_bleu
        }

    # Hypothesis Testing with Holm-Bonferroni Correction:
    # H1: M3 ROUGE-Lsum > M1 ROUGE-Lsum
    # H2: M3 BLEU > M1 BLEU
    # H3: M3 ROUGE-Lsum > M2 ROUGE-Lsum
    # H4: M3 ROUGE-Lsum > Control_A ROUGE-Lsum
    hypotheses = [
        ("H1_M3_vs_M1_ROUGEL", variant_per_case_metrics["M3"]["rougeLsum"], variant_per_case_metrics["M1"]["rougeLsum"]),
        ("H2_M3_vs_M1_BLEU",   variant_per_case_metrics["M3"]["bleu"],      variant_per_case_metrics["M1"]["bleu"]),
        ("H3_M3_vs_M2_ROUGEL", variant_per_case_metrics["M3"]["rougeLsum"], variant_per_case_metrics["M2"]["rougeLsum"]),
        ("H4_M3_vs_CtrlA_ROUGEL", variant_per_case_metrics["M3"]["rougeLsum"], variant_per_case_metrics["Control_A"]["rougeLsum"])
    ]

    raw_p_values = []
    test_reports = []
    for h_name, s_a, s_b in hypotheses:
        b_res = paired_bootstrap_test(s_a, s_b, n_resamples=1000, seed=config.seed)
        raw_p_values.append(b_res["p_value"])
        test_reports.append({
            "hypothesis": h_name,
            "mean_difference": b_res["mean_diff"] * 100.0,
            "ci_95": [b_res["ci_lower"] * 100.0, b_res["ci_upper"] * 100.0],
            "raw_p_value": b_res["p_value"]
        })

    adj_p_values = holm_bonferroni_correction(raw_p_values)
    for report, adj_p in zip(test_reports, adj_p_values):
        report["adjusted_p_value"] = adj_p
        report["significant_at_05"] = bool(adj_p < 0.05)

    # Save official submission file for M3
    safe_name = model_name.replace("/", "_").replace("-", "_")
    sub_path = config.output_dir / f"submission_{safe_name}.json"
    official_submission = [
        {"case_id": p["case_id"], "answer": p["answer"]}
        for p in variants["M3"]
    ]
    with open(sub_path, "w", encoding="utf-8") as f:
        json.dump(official_submission, f, indent=2)

    print(f"\nOfficial submission saved to: {sub_path}")

    # Summary table
    print("\n--- Test Set Evaluation Ladder ---")
    print(f"{'Variant':<12} | {'ROUGE-Lsum':<12} | {'BLEU':<10} | {'Overall Score':<14}")
    print("-" * 55)
    for var in ["M0", "M1", "Control_A", "Control_B", "M2", "M3"]:
        res = variant_results[var]
        print(f"{var:<12} | {res.get('rougeLsum', 0.0):<12.2f} | {res.get('bleu', 0.0):<10.2f} | {res.get('overall_relevance_score', 0.0):<14.2f}")

    print("\n--- Paired Bootstrap Hypothesis Tests (Holm-Bonferroni corrected) ---")
    for r in test_reports:
        sig_mark = "*** (p < 0.05)" if r["significant_at_05"] else "(n.s.)"
        print(f"  {r['hypothesis']:<25}: Diff = +{r['mean_difference']:.2f} pts "
              f"[95% CI: {r['ci_95'][0]:.2f}, {r['ci_95'][1]:.2f}], "
              f"adj-p = {r['adjusted_p_value']:.4f} {sig_mark}")

    result_payload = {
        "model_name": model_name,
        "ablation_ladder": variant_results,
        "hypothesis_tests": test_reports,
        "submission_file": str(sub_path)
    }

    if save_rollouts:
        rollout_path = config.output_dir / f"rollouts_{safe_name}.json"
        with open(rollout_path, "w", encoding="utf-8") as f:
            json.dump(all_rollouts, f, indent=2)
        result_payload["rollout_file"] = str(rollout_path)

    return result_payload


def main():
    parser = argparse.ArgumentParser(description="ArchEHR-QA Subtask 3 Test Benchmark Runner")
    parser.add_argument("--models", type=str, default="Qwen/Qwen2.5-3B-Instruct",
                        help="Comma-separated model names or 'all' or 'slms' or 'llms'")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit number of test cases (for smoke testing)")
    parser.add_argument("--output_dir", type=str, default="outputs",
                        help="Output directory for predictions and metrics")
    parser.add_argument("--save_rollouts", action="store_true",
                        help="Save raw stochastic rollouts to disk")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    args = parser.parse_args()

    config = PipelineConfig()
    config.seed = args.seed
    config.output_dir = Path(args.output_dir)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # Determine models to evaluate
    if args.models == "all":
        models_to_run = DEFAULT_MODELS["slms"] + DEFAULT_MODELS["llms"]
    elif args.models == "slms":
        models_to_run = DEFAULT_MODELS["slms"]
    elif args.models == "llms":
        models_to_run = DEFAULT_MODELS["llms"]
    else:
        models_to_run = [m.strip() for m in args.models.split(",") if m.strip()]

    # Load test cases
    test_cases = parse_cases_from_xml(config.data_root / "test", with_key=True)
    if args.limit:
        test_cases = test_cases[:args.limit]
        print(f"Smoke-test mode: limited to {len(test_cases)} cases.")

    print(f"Loaded {len(test_cases)} test cases from {config.data_root / 'test'}")

    all_model_results = {}
    for model_name in models_to_run:
        res = evaluate_test_model(
            model_name=model_name,
            test_cases=test_cases,
            config=config,
            save_rollouts=args.save_rollouts
        )
        all_model_results[model_name] = res

    # Save complete benchmark summary
    summary_path = config.output_dir / "test_benchmark_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_model_results, f, indent=2)

    print(f"\n=======================================================")
    print(f"Complete test benchmark saved to: {summary_path}")
    print(f"=======================================================")


if __name__ == "__main__":
    main()
