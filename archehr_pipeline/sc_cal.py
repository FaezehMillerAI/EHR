"""
Calibrated Sample-Consistency (SC-Cal) for ArchEHR-QA.
Implements:
1. MBR medoid consensus answer selection over stochastic rollouts.
2. SelfCheckGPT-style sentence-level consistency scoring c(s_k).
3. Citation stripping prior to embedding.
4. Platt scaling (2-parameter logistic calibration).
5. AUROC, ECE (10 bins), Brier Score, and 1,000 bootstrap confidence intervals.
"""

import re
import math
from typing import List, Dict, Tuple, Optional, Any
try:
    import numpy as np
    _HAS_NUMPY = True
except ImportError:
    np = None
    _HAS_NUMPY = False

from archehr_pipeline.config import PipelineConfig
from archehr_pipeline.generator import strip_citations

try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except ImportError:
    SentenceTransformer = None
    _HAS_ST = False

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score, brier_score_loss
    _HAS_SKLEARN = True
except ImportError:
    _HAS_SKLEARN = False


def extract_sentences_with_citations(answer_text: str) -> List[Dict[str, Any]]:
    """
    Parse answer into sentences and their associated pipe citations.
    Official format: 'Sentence text. |id1, id2|'
    """
    if not answer_text or not answer_text.strip():
        return []

    lines = [line.strip() for line in answer_text.strip().split("\n") if line.strip()]
    if not lines:
        lines = [s.strip() for s in re.split(r"(?<=[.!?])\s+", answer_text.strip()) if s.strip()]

    parsed_sentences = []
    for line in lines:
        # Match pipe citations: e.g. |2| or |2, 6|
        pipe_match = re.search(r"\|([^|]+)\|", line)
        if pipe_match:
            raw_cites = pipe_match.group(1)
            citations = [c.strip() for c in re.findall(r"\d+", raw_cites)]
            sent_text = line[:pipe_match.start()].strip()
        else:
            # Check bracket citations as fallback: e.g. [2]
            bracket_match = re.search(r"\[([\d\s,]+)\]", line)
            if bracket_match:
                citations = [c.strip() for c in re.findall(r"\d+", bracket_match.group(1))]
                sent_text = line[:bracket_match.start()].strip()
            else:
                citations = []
                sent_text = line.strip()

        # Add period if missing
        if sent_text and sent_text[-1] not in ".!?":
            sent_text += "."

        if sent_text:
            parsed_sentences.append({
                "sentence_text": sent_text,
                "clean_text": strip_citations(sent_text),
                "citations": citations,
                "raw_line": line
            })

    return parsed_sentences


def compute_cosine_similarity_matrix(embeddings: Any) -> Any:
    """Compute normalized pairwise cosine similarity matrix."""
    if _HAS_NUMPY and hasattr(embeddings, "shape"):
        norm = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norm = np.where(norm == 0, 1e-12, norm)
        normalized = embeddings / norm
        return np.clip(normalized @ normalized.T, -1.0, 1.0)
    else:
        matrix = []
        for v1 in embeddings:
            row = []
            norm1 = math.sqrt(sum(x * x for x in v1)) or 1e-12
            for v2 in embeddings:
                norm2 = math.sqrt(sum(y * y for y in v2)) or 1e-12
                dot = sum(x * y for x, y in zip(v1, v2))
                cos = max(-1.0, min(1.0, dot / (norm1 * norm2)))
                row.append(cos)
            matrix.append(row)
        return matrix


class SampleConsistencyCalibrator:
    """
    SC-Cal: Computes rollout consensus, selects medoid answer,
    evaluates sentence-level consistency, and fits Platt calibration.
    """
    def __init__(self, config: Optional[PipelineConfig] = None, embedder_name: Optional[str] = None):
        self.config = config or PipelineConfig()
        self.embedder_name = embedder_name or self.config.embedding_model
        self.embedder = None
        self.platt_model: Optional[Any] = None  # LogisticRegression (slope w, intercept b)

    def load_embedder(self):
        if not _HAS_ST:
            return
        if self.embedder is None:
            try:
                self.embedder = SentenceTransformer(self.embedder_name, device=self.config.device)
            except Exception:
                # Fallback to general embedding model if biomedical model download fails
                self.embedder = SentenceTransformer(self.config.embedding_fallback, device=self.config.device)

    def encode_texts(self, texts: List[str]) -> Any:
        """Encode list of texts into dense vectors with citations stripped."""
        clean = [strip_citations(t) for t in texts]
        if not _HAS_ST or self.embedder is None:
            # Deterministic bag-of-words character hash mock embedding for testing without SentenceTransformer
            vecs = []
            for t in clean:
                v = np.zeros(64, dtype=np.float32) if _HAS_NUMPY else [0.0] * 64
                words = re.findall(r"\w+", t.lower())
                for w in words:
                    v[hash(w) % 64] += 1.0
                norm = math.sqrt(sum(x * x for x in v))
                norm = norm if norm > 0 else 1.0
                if _HAS_NUMPY:
                    vecs.append(v / norm)
                else:
                    vecs.append([x / norm for x in v])
            return np.stack(vecs) if _HAS_NUMPY else vecs

        self.load_embedder()
        embs = self.embedder.encode(clean, normalize_embeddings=True, show_progress_bar=False)
        return np.array(embs) if _HAS_NUMPY else embs

    def select_medoid(self, rollouts: List[str]) -> Tuple[int, str, float, Any]:
        """
        Minimum Bayes Risk (MBR) medoid selection:
        Finds the rollout that maximizes cosine agreement across all rollouts.
        Returns:
          medoid_idx, medoid_answer, answer_consensus_score, similarity_matrix
        """
        if not rollouts:
            return 0, "", 0.0, None
        if len(rollouts) == 1:
            return 0, rollouts[0], 1.0, None

        # Citations are stripped inside encode_texts to prevent citation ID artifacts
        embs = self.encode_texts(rollouts)
        sim_matrix = compute_cosine_similarity_matrix(embs)

        # Medoid maximizes average similarity to other rollouts
        if _HAS_NUMPY and hasattr(sim_matrix, "sum"):
            row_sums = sim_matrix.sum(axis=1)
            medoid_idx = int(np.argmax(row_sums))
            off_diag_sum = float(sim_matrix.sum() - np.trace(sim_matrix))
        else:
            row_sums = [sum(row) for row in sim_matrix]
            medoid_idx = int(max(range(len(row_sums)), key=lambda i: row_sums[i]))
            trace = sum(sim_matrix[i][i] for i in range(len(sim_matrix)))
            off_diag_sum = sum(sum(row) for row in sim_matrix) - trace

        medoid_answer = rollouts[medoid_idx]
        r = len(rollouts)
        answer_consensus = float(max(0.0, min(1.0, off_diag_sum / (r * (r - 1)))))

        return medoid_idx, medoid_answer, answer_consensus, sim_matrix

    def compute_sentence_consistency(
        self,
        medoid_answer: str,
        rollouts: List[str],
        medoid_idx: int
    ) -> List[Dict[str, Any]]:
        """
        SelfCheckGPT-style sentence-level consistency scoring:
        For each sentence s_k in the medoid, compute its average maximum semantic similarity
        across the sentences of the other R-1 rollouts:
          c(s_k) = (1 / (R - 1)) * sum_{j != medoid} max_{s' in r_j} cos(e(s_k), e(s'))
        """
        medoid_sentences = extract_sentences_with_citations(medoid_answer)
        if not medoid_sentences or len(rollouts) <= 1:
            for s in medoid_sentences:
                s["raw_consistency"] = 1.0
                s["calibrated_prob"] = 1.0
            return medoid_sentences

        # Decompose other rollouts into sentences
        other_rollout_sentences: List[List[Dict[str, Any]]] = []
        for i, r in enumerate(rollouts):
            if i == medoid_idx:
                continue
            r_sents = extract_sentences_with_citations(r)
            if r_sents:
                other_rollout_sentences.append(r_sents)

        if not other_rollout_sentences:
            for s in medoid_sentences:
                s["raw_consistency"] = 1.0
                s["calibrated_prob"] = 1.0
            return medoid_sentences

        # Pre-embed medoid sentences
        medoid_texts = [s["clean_text"] for s in medoid_sentences]
        medoid_embs = self.encode_texts(medoid_texts)

        # Pre-embed all sentences in other rollouts
        all_other_sents = [s["clean_text"] for r_sents in other_rollout_sentences for s in r_sents]
        other_embs = self.encode_texts(all_other_sents)

        # Map back to rollout clusters
        rollout_emb_chunks = []
        curr = 0
        for r_sents in other_rollout_sentences:
            sz = len(r_sents)
            rollout_emb_chunks.append(other_embs[curr:curr + sz])
            curr += sz

        # Compute consistency c(s_k) for each medoid sentence
        for k_idx, s in enumerate(medoid_sentences):
            e_k = medoid_embs[k_idx]
            max_sims = []
            for chunk in rollout_emb_chunks:
                sims = []
                norm_k = math.sqrt(sum(x * x for x in e_k)) or 1e-12
                for e_other in chunk:
                    norm_o = math.sqrt(sum(y * y for y in e_other)) or 1e-12
                    dot = sum(x * y for x, y in zip(e_k, e_other))
                    sims.append(max(-1.0, min(1.0, dot / (norm_k * norm_o))))
                max_sims.append(max(sims) if sims else 0.0)

            raw_c = float(sum(max_sims) / len(max_sims)) if max_sims else 0.0
            s["raw_consistency"] = raw_c
            s["calibrated_prob"] = self.apply_platt_scaling(raw_c)

        return medoid_sentences

    def fit_platt_scaling(self, scores: List[float], binary_labels: List[int]):
        """
        Fit Platt scaling (2-parameter logistic regression: slope w, intercept b)
        on out-of-fold dev calibration samples:
          P(y=1 | c) = 1 / (1 + exp(-(w * c + b)))
        """
        if not scores or len(set(binary_labels)) < 2:
            return

        if _HAS_SKLEARN and _HAS_NUMPY:
            scores_arr = np.array(scores).reshape(-1, 1)
            labels_arr = np.array(binary_labels)
            clf = LogisticRegression(penalty=None, solver="lbfgs")
            clf.fit(scores_arr, labels_arr)
            self.platt_model = clf
        else:
            # Analytical slope/intercept estimate fallback
            pos_scores = [s for s, y in zip(scores, binary_labels) if y == 1]
            neg_scores = [s for s, y in zip(scores, binary_labels) if y == 0]
            mean_pos = sum(pos_scores) / len(pos_scores) if pos_scores else 0.5
            mean_neg = sum(neg_scores) / len(neg_scores) if neg_scores else 0.5
            w = 5.0 if mean_pos >= mean_neg else -5.0
            b = -w * (mean_pos + mean_neg) / 2.0
            self.platt_model = (w, b)

    def apply_platt_scaling(self, score: float) -> float:
        """Map raw consistency score c to calibrated probability in [0, 1]."""
        if self.platt_model is None:
            # If not yet calibrated, clip raw cosine to [0, 1]
            return float(max(0.0, min(1.0, score)))

        if _HAS_SKLEARN and hasattr(self.platt_model, "predict_proba"):
            p = float(self.platt_model.predict_proba([[score]])[0, 1])
            return p
        else:
            w, b = self.platt_model
            logit = w * score + b
            return 1.0 / (1.0 + math.exp(-logit))


def compute_calibration_metrics(
    confidences: List[float],
    binary_labels: List[int],
    num_bins: int = 10,
    n_bootstrap: int = 1000,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Compute AUROC, Expected Calibration Error (ECE, 10 bins), and Brier Score
    with 1,000 bootstrap resamples for 95% confidence intervals.
    """
    confs = list(confidences)
    labels = list(binary_labels)
    n = len(confs)

    if n == 0 or len(set(labels)) < 2:
        return {
            "auroc": 0.5,
            "ece": 0.0,
            "brier": 0.0,
            "auroc_ci": (0.5, 0.5),
            "ece_ci": (0.0, 0.0),
            "brier_ci": (0.0, 0.0)
        }

    def calc_metrics(c_list, y_list):
        # AUROC via Mann-Whitney U / concordance
        pos = [c for c, y in zip(c_list, y_list) if y == 1]
        neg = [c for c, y in zip(c_list, y_list) if y == 0]
        if not pos or not neg:
            auc = 0.5
        elif _HAS_SKLEARN:
            auc = float(roc_auc_score(y_list, c_list))
        else:
            pairs = 0.0
            for p in pos:
                for q in neg:
                    if p > q:
                        pairs += 1.0
                    elif p == q:
                        pairs += 0.5
            auc = float(pairs / (len(pos) * len(neg)))

        # Brier Score
        brier = float(sum((c - y) ** 2 for c, y in zip(c_list, y_list)) / len(c_list))

        # Expected Calibration Error (ECE)
        bin_boundaries = [i / num_bins for i in range(num_bins + 1)]
        ece = 0.0
        for i in range(num_bins):
            bin_lower = bin_boundaries[i]
            bin_upper = bin_boundaries[i + 1]
            if i < num_bins - 1:
                in_bin_indices = [idx for idx, c in enumerate(c_list) if bin_lower <= c < bin_upper]
            else:
                in_bin_indices = [idx for idx, c in enumerate(c_list) if bin_lower <= c <= bin_upper]
            bin_size = len(in_bin_indices)
            if bin_size > 0:
                acc = sum(y_list[idx] for idx in in_bin_indices) / bin_size
                conf = sum(c_list[idx] for idx in in_bin_indices) / bin_size
                ece += (bin_size / len(c_list)) * abs(acc - conf)

        return auc, float(ece), brier

    main_auc, main_ece, main_brier = calc_metrics(confs, labels)

    # Bootstrap resamples
    import random
    rng = random.Random(seed)
    boot_aucs, boot_eces, boot_briers = [], [], []
    for _ in range(n_bootstrap):
        sampled_indices = [rng.randint(0, n - 1) for _ in range(n)]
        s_confs = [confs[i] for i in sampled_indices]
        s_labels = [labels[i] for i in sampled_indices]
        b_auc, b_ece, b_brier = calc_metrics(s_confs, s_labels)
        boot_aucs.append(b_auc)
        boot_eces.append(b_ece)
        boot_briers.append(b_brier)

    boot_aucs.sort()
    boot_eces.sort()
    boot_briers.sort()

    low_idx = int(0.025 * n_bootstrap)
    high_idx = int(0.975 * n_bootstrap)

    return {
        "auroc": main_auc,
        "ece": main_ece,
        "brier": main_brier,
        "auroc_ci": (float(boot_aucs[low_idx]), float(boot_aucs[high_idx])),
        "ece_ci": (float(boot_eces[low_idx]), float(boot_eces[high_idx])),
        "brier_ci": (float(boot_briers[low_idx]), float(boot_briers[high_idx]))
    }
