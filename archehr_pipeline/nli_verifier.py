"""
Independent NLI-Guided Evidence Attribution & Selective Hallucination Pruning (NLI-SHP).
Verifies claim-to-evidence entailment using decoupled NLI models and multi-sentence premise concatenation.
"""

from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import re

from archehr_pipeline.config import PipelineConfig
from archehr_pipeline.data_loader import Case
from archehr_pipeline.sc_cal import extract_sentences_with_citations

try:
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    _HAS_TORCH_HF = True
except ImportError:
    torch = None
    _HAS_TORCH_HF = False


def build_concatenated_premise(citation_ids: List[str], sentence_map: Dict[str, str]) -> str:
    """
    Concatenate note sentences cited in multi-evidence citations (e.g. |2, 6|)
    so that multi-hop / composite clinical claims can be fully entailed.
    """
    texts = [sentence_map[cid] for cid in citation_ids if cid in sentence_map]
    return " ".join(texts) if texts else ""


class NLIClaimVerifier:
    """
    NLI Attribution & Pruning engine with decoupled models:
      - Runtime Pruner: cross-encoder/nli-deberta-v3-small
      - Ground-truth Labeler: MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli
    """
    def __init__(
        self,
        config: Optional[PipelineConfig] = None,
        pruner_model_name: Optional[str] = None,
        labeler_model_name: Optional[str] = None
    ):
        self.config = config or PipelineConfig()
        self.pruner_model_name = pruner_model_name or self.config.pruner_nli_model
        self.labeler_model_name = labeler_model_name or self.config.labeler_nli_model

        self.pruner_tok = None
        self.pruner_model = None

        self.labeler_tok = None
        self.labeler_model = None

    def load_pruner(self):
        if not _HAS_TORCH_HF:
            return
        if self.pruner_model is None:
            self.pruner_tok = AutoTokenizer.from_pretrained(self.pruner_model_name)
            self.pruner_model = AutoModelForSequenceClassification.from_pretrained(
                self.pruner_model_name
            ).to(self.config.device).eval()

    def load_labeler(self):
        if not _HAS_TORCH_HF:
            return
        if self.labeler_model is None:
            self.labeler_tok = AutoTokenizer.from_pretrained(self.labeler_model_name)
            self.labeler_model = AutoModelForSequenceClassification.from_pretrained(
                self.labeler_model_name
            ).to(self.config.device).eval()

    @torch.no_grad() if _HAS_TORCH_HF else lambda fn: fn
    def score_entailment(
        self,
        premise: str,
        hypothesis: str,
        use_labeler: bool = False
    ) -> float:
        """
        Compute NLI entailment probability P(premise entails hypothesis).
        """
        if not premise or not hypothesis:
            return 0.0

        if not _HAS_TORCH_HF:
            # Word-overlap heuristic fallback when testing without PyTorch
            p_words = set(re.findall(r"\w+", premise.lower()))
            h_words = set(re.findall(r"\w+", hypothesis.lower()))
            return float(len(p_words & h_words) / max(1, len(h_words)))

        if use_labeler:
            self.load_labeler()
            tok, model = self.labeler_tok, self.labeler_model
        else:
            self.load_pruner()
            tok, model = self.pruner_tok, self.pruner_model

        inputs = tok(
            premise,
            hypothesis,
            truncation=True,
            max_length=512,
            return_tensors="pt"
        ).to(self.config.device)

        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().tolist()

        # In standard 3-class NLI: [entailment, neutral, contradiction] or [contradiction, neutral, entailment]
        # DeBERTa-v3 NLI mapping: index 0 or index 2 depending on id2label
        entailment_idx = 0
        if hasattr(model.config, "id2label"):
            for idx, label_name in model.config.id2label.items():
                if "entail" in label_name.lower():
                    entailment_idx = int(idx)
                    break
        elif len(probs) == 3:
            entailment_idx = 0 if probs[0] > probs[2] else 2

        return float(probs[entailment_idx])

    def evaluate_ground_truth_sentence_labels(
        self,
        medoid_sentences: List[Dict[str, Any]],
        case: Case,
        strict: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Compute ground-truth factual correctness label y in {0, 1}
        strictly on the pre-pruning medoid sentences:
          y = 1 iff:
            1. Valid citation: C(s_k) intersects gold essential (strict) or essential U supplementary (lenient)
            2. Premise entailment: P(s_k | concatenated_premise) >= 0.50 under independent large labeler
        """
        gold_citations = set(case.essential_sentence_ids if strict else case.lenient_sentence_ids)
        sent_map = case.sentence_map

        for s in medoid_sentences:
            citations = set(s.get("citations", []))
            has_valid_citation = bool(citations & gold_citations)

            if not citations or not has_valid_citation:
                premise_text = ""
                p_entail = 0.0
                citation_error = True
                semantic_hallucination = False
                is_correct = 0
            else:
                citation_error = False
                premise_text = build_concatenated_premise(list(citations), sent_map)
                p_entail = self.score_entailment(premise_text, s["clean_text"], use_labeler=True)
                if p_entail >= 0.50:
                    semantic_hallucination = False
                    is_correct = 1
                else:
                    semantic_hallucination = True
                    is_correct = 0

            s["gold_y"] = is_correct
            s["citation_error"] = citation_error
            s["semantic_hallucination"] = semantic_hallucination
            s["labeler_entailment"] = p_entail

        return medoid_sentences

    def verify_and_prune(
        self,
        medoid_sentences: List[Dict[str, Any]],
        case: Case,
        context_sentences: List[Dict[str, Any]],
        theta_ungrounded: Optional[float] = None
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        NLI-SHP:
        1. Evaluates each medoid sentence against its cited premise.
        2. Prunes sentences where entailment < theta_ungrounded.
        3. Fallback: If all sentences are pruned, retains the single sentence with highest entailment.
        4. Re-formats answer as 'Sentence text. |id1, id2|'.
        """
        theta_ung = theta_ungrounded if theta_ungrounded is not None else self.config.theta_ungrounded
        sent_map = case.sentence_map

        scored_sentences = []
        for s in medoid_sentences:
            cites = s.get("citations", [])
            if cites:
                premise = build_concatenated_premise(cites, sent_map)
            else:
                # If no citation, check against full selected context
                premise = " ".join([cs["text"] for cs in context_sentences[:3]])

            p_entail = self.score_entailment(premise, s["clean_text"], use_labeler=False)
            s_copy = dict(s)
            s_copy["pruner_entailment"] = p_entail
            scored_sentences.append(s_copy)

        # Filter sentences exceeding threshold
        retained = [s for s in scored_sentences if s["pruner_entailment"] >= theta_ung]

        # Empty answer fallback: keep sentence with highest entailment
        if not retained and scored_sentences:
            best_s = max(scored_sentences, key=lambda x: x["pruner_entailment"])
            retained = [best_s]

        # Build output answer string with official pipe citations
        output_lines = []
        for s in retained:
            cites = s.get("citations", [])
            cite_str = f" |{', '.join(cites)}|" if cites else ""
            txt = s["clean_text"]
            if txt and txt[-1] not in ".!?":
                txt += "."
            output_lines.append(f"{txt}{cite_str}")

        final_answer = "\n".join(output_lines)
        return final_answer, retained
