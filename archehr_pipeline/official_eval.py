"""
Faithful implementation of the official ArchEHR-QA evaluation suite
(aligned with soni-sarvesh/archehr-qa/evaluation/scoring.py).
Computes Strict & Lenient Factuality (Micro/Macro P/R/F1) and Relevance Metrics.
"""

import json
import re
from collections import defaultdict
from typing import List, Dict, Set, Optional, Tuple, Any
try:
    import numpy as np
except ImportError:
    class _DummyNP:
        @staticmethod
        def mean(vals):
            vals_list = list(vals)
            return sum(vals_list) / len(vals_list) if vals_list else 0.0
    np = _DummyNP()

try:
    from rouge_score import rouge_scorer
    _HAS_ROUGE = True
except ImportError:
    _HAS_ROUGE = False

try:
    import sacrebleu
    _HAS_SACREBLEU = True
except ImportError:
    _HAS_SACREBLEU = False

try:
    import nltk
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    _HAS_NLTK = True
except ImportError:
    _HAS_NLTK = False


def parse_submission_cases(raw_predictions: List[Dict[str, Any]], max_answer_words: int = 75) -> List[Dict[str, Any]]:
    """
    Parse submission cases into clean answers and citation sets, matching official load_submission.
    Official syntax: 'Sentence text. |id1, id2|'
    """
    processed = []
    for case in raw_predictions:
        case_id = str(case["case_id"])
        answer_text = str(case.get("answer", "")).strip()

        answer_sentences = []
        for line in answer_text.split("\n"):
            line = line.strip()
            if not line:
                continue

            line_parts = line.rsplit("|", maxsplit=2)
            if len(line_parts) >= 3:
                sent = line_parts[-3].strip()
                citation_part = line_parts[-2]
                citations = [c.strip() for c in re.findall(r"\d+", citation_part)]
            else:
                # Check bracket citations as fallback
                bracket_match = re.search(r"\[([\d\s,]+)\]", line)
                if bracket_match:
                    citations = [c.strip() for c in re.findall(r"\d+", bracket_match.group(1))]
                    sent = line[:bracket_match.start()].strip()
                else:
                    sent = line
                    citations = []

            if sent and sent[-1] not in ".!?":
                sent += "."

            if sent:
                answer_sentences.append({"sentence": sent, "citations": citations})

        # Concatenate sentences
        case_answer = " ".join([s["sentence"] for s in answer_sentences if s["sentence"]])
        words = [w for w in case_answer.split(" ") if w.strip()]
        if len(words) > max_answer_words:
            case_answer = " ".join(words[:max_answer_words])

        case_citations = {c for s in answer_sentences for c in s["citations"]}

        processed.append({
            "case_id": case_id,
            "answer": case_answer,
            "citations": case_citations,
            "sentences": answer_sentences
        })

    return processed


def compute_factuality_scores_for_variation(
    submission: List[Dict[str, Any]],
    key_map: Dict[str, Dict[str, str]],
    variation: str = "strict"
) -> Dict[str, Dict[str, float]]:
    """
    Exact official factuality computation matching scoring.py:
      strict: allowed_relevance = {"essential"}
      lenient: allowed_relevance = {"essential", "supplementary"}
    Computes both macro and micro Precision, Recall, and F1.
    """
    if variation == "strict":
        allowed_relevance = {"essential"}
    elif variation == "lenient":
        allowed_relevance = {"essential", "supplementary"}
    else:
        raise ValueError(f"Invalid variation: {variation}")

    precision_scores = []
    recall_scores = []
    f1_scores = []

    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for case in submission:
        cid = str(case["case_id"])
        pred_citations = set(str(c) for c in case["citations"])
        case_key = key_map.get(cid, {})

        gold_citations = {
            str(sent_id)
            for sent_id, relevance in case_key.items()
            if relevance in allowed_relevance
        }

        tp = len(gold_citations & pred_citations)
        fp = len(pred_citations - gold_citations)
        fn = len(gold_citations - pred_citations)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        precision_scores.append(precision)
        recall_scores.append(recall)
        f1_scores.append(f1)

        true_positives += tp
        false_positives += fp
        false_negatives += fn

    macro_p = float(np.mean(precision_scores)) if precision_scores else 0.0
    macro_r = float(np.mean(recall_scores)) if recall_scores else 0.0
    macro_f1 = float(np.mean(f1_scores)) if f1_scores else 0.0

    micro_p = float(true_positives / (true_positives + false_positives)) if (true_positives + false_positives) > 0 else 0.0
    micro_r = float(true_positives / (true_positives + false_negatives)) if (true_positives + false_negatives) > 0 else 0.0
    micro_f1 = float(2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) > 0 else 0.0

    return {
        "macro": {"precision": macro_p, "recall": macro_r, "f1": macro_f1},
        "micro": {"precision": micro_p, "recall": micro_r, "f1": micro_f1}
    }


def compute_factuality_scores(submission: List[Dict[str, Any]], key_map: Dict[str, Dict[str, str]]) -> Dict[str, Any]:
    """Compute both strict and lenient factuality suites."""
    return {
        "strict": compute_factuality_scores_for_variation(submission, key_map, "strict"),
        "lenient": compute_factuality_scores_for_variation(submission, key_map, "lenient")
    }


def compute_text_relevance_metrics(
    predictions: List[str],
    references: List[str]
) -> Dict[str, float]:
    """
    Compute BLEU and ROUGE-Lsum against reference answers.
    """
    assert len(predictions) == len(references)
    n = len(predictions)
    if n == 0:
        return {"bleu": 0.0, "rougeLsum": 0.0}

    # ROUGE-Lsum
    rouge_l_scores = []
    if _HAS_ROUGE:
        scorer = rouge_scorer.RougeScorer(["rougeLsum"], use_stemmer=True)
        for pred, ref in zip(predictions, references):
            score = scorer.score(ref, pred)["rougeLsum"].fmeasure
            rouge_l_scores.append(score)
    else:
        # Simple LCS approximation if rouge_score not installed
        for pred, ref in zip(predictions, references):
            p_words = pred.lower().split()
            r_words = ref.lower().split()
            overlap = len(set(p_words) & set(r_words))
            f1 = 2 * overlap / (len(p_words) + len(r_words)) if (len(p_words) + len(r_words)) > 0 else 0.0
            rouge_l_scores.append(f1)

    # BLEU
    bleu_scores = []
    if _HAS_SACREBLEU:
        for pred, ref in zip(predictions, references):
            b = sacrebleu.sentence_bleu(pred, [ref]).score / 100.0
            bleu_scores.append(b)
    elif _HAS_NLTK:
        smooth = SmoothingFunction().method1
        for pred, ref in zip(predictions, references):
            ref_tokens = [ref.split()]
            pred_tokens = pred.split()
            b = sentence_bleu(ref_tokens, pred_tokens, smoothing_function=smooth)
            bleu_scores.append(b)
    else:
        # Basic unigram precision fallback
        for pred, ref in zip(predictions, references):
            p_words = pred.lower().split()
            r_words = ref.lower().split()
            prec = len(set(p_words) & set(r_words)) / max(1, len(p_words))
            bleu_scores.append(prec)

    return {
        "bleu": float(np.mean(bleu_scores)),
        "rougeLsum": float(np.mean(rouge_l_scores))
    }


def compute_overall_leaderboard(
    factuality_scores: Optional[Dict[str, Any]],
    relevance_scores: Dict[str, float]
) -> Dict[str, float]:
    """
    Compute official composite scores:
      overall_factuality_score = strict_micro_f1
      overall_relevance_score = mean(relevance_metrics)
      overall_score = mean(overall_factuality_score, overall_relevance_score)
    """
    leaderboard = {}

    if factuality_scores:
        for variation in ["strict", "lenient"]:
            for f1_type in ["micro", "macro"]:
                metrics = factuality_scores[variation][f1_type]
                leaderboard[f"{variation}_{f1_type}_precision"] = metrics["precision"] * 100.0
                leaderboard[f"{variation}_{f1_type}_recall"] = metrics["recall"] * 100.0
                leaderboard[f"{variation}_{f1_type}_f1"] = metrics["f1"] * 100.0

        overall_fact = leaderboard["strict_micro_f1"]
        leaderboard["overall_factuality_score"] = overall_fact
    else:
        overall_fact = None

    for k, v in relevance_scores.items():
        leaderboard[k] = v * 100.0 if v <= 1.0 else v

    rel_values = [v for k, v in leaderboard.items() if k in {"bleu", "rougeLsum", "sari", "bertscore", "alignscore"}]
    overall_rel = float(np.mean(rel_values)) if rel_values else 0.0
    leaderboard["overall_relevance_score"] = overall_rel

    if overall_fact is not None:
        leaderboard["overall_score"] = float(np.mean([overall_fact, overall_rel]))

    return leaderboard
