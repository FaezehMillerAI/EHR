"""
Multi-Perspective Graded Sentence Ranking (MPC-GR) for ArchEHR-QA.
Scores and reranks clinical note sentences using a fine-tuned biomedical cross-encoder.
"""

from typing import List, Dict, Tuple, Optional, Any
from pathlib import Path
import re

from archehr_pipeline.config import PipelineConfig
from archehr_pipeline.data_loader import Case

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    _HAS_TORCH_HF = True
except ImportError:
    torch = None
    nn = None
    _HAS_TORCH_HF = False


def format_cross_encoder_input(
    clinician_q: str,
    patient_q: str,
    narrative: str,
    sentence_text: str,
    perspective_mode: str = "P4"
) -> Tuple[str, str]:
    """
    Construct cross-encoder pair (text_a, text_b) safely respecting 512 token limits.
    To ensure the candidate sentence is never pushed out by long narratives,
    the candidate sentence is placed in the primary position.

    Perspective Modes:
      - P1: Patient question only
      - P2: Clinician question only
      - P3: Patient question + Clinician question
      - P4: Full multi-perspective (Clinician question + Candidate sentence + Narrative[:300 chars])
    """
    sentence_clean = re.sub(r"\s+", " ", sentence_text or "").strip()
    clin_clean = re.sub(r"\s+", " ", clinician_q or "").strip()
    pat_clean = re.sub(r"\s+", " ", patient_q or "").strip()
    narr_clean = re.sub(r"\s+", " ", narrative or "").strip()[:300]

    if perspective_mode == "P1":
        query_text = f"Patient Question: {pat_clean}"
        doc_text = sentence_clean
    elif perspective_mode == "P2":
        query_text = f"Clinician Question: {clin_clean}"
        doc_text = sentence_clean
    elif perspective_mode == "P3":
        query_text = f"Clinician: {clin_clean} | Patient: {pat_clean}"
        doc_text = sentence_clean
    else:  # P4: Full
        query_text = f"Clinician: {clin_clean} | Patient: {pat_clean}"
        doc_text = f"Sentence: {sentence_clean} | Narrative: {narr_clean}"

    return query_text, doc_text


if _HAS_TORCH_HF:
    class GradedSentenceDataset(Dataset):
        def __init__(self, examples: List[Dict[str, Any]], tokenizer, max_length: int = 512, perspective_mode: str = "P4"):
            self.examples = examples
            self.tokenizer = tokenizer
            self.max_length = max_length
            self.mode = perspective_mode

        def __len__(self):
            return len(self.examples)

        def __getitem__(self, idx):
            ex = self.examples[idx]
            qa, qb = format_cross_encoder_input(
                clinician_q=ex["clinician_question"],
                patient_q=ex["patient_question"],
                narrative=ex["narrative"],
                sentence_text=ex["sentence_text"],
                perspective_mode=self.mode
            )
            encoded = self.tokenizer(
                qa,
                qb,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )
            item = {k: v.squeeze(0) for k, v in encoded.items()}
            # Graded target: 0, 1, 2
            item["labels"] = torch.tensor(ex["label_graded"], dtype=torch.long)
            return item


class MultiPerspectiveSentenceRanker:
    """
    Cross-Encoder Sentence Ranker supporting 3-tier graded labels
    (essential=2, supplementary=1, not-relevant=0) and multi-perspective templating.
    """
    def __init__(self, config: Optional[PipelineConfig] = None, model_name_or_path: Optional[str] = None):
        self.config = config or PipelineConfig()
        self.model_name = model_name_or_path or self.config.ranker_base_model
        self.device = self.config.device
        self.tokenizer = None
        self.model = None

    def load_model(self):
        if not _HAS_TORCH_HF:
            raise RuntimeError("PyTorch and HuggingFace Transformers are required to load the cross-encoder model.")
        if self.model is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            # 3-class classification head for graded relevance: 0=not-relevant, 1=supplementary, 2=essential
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                num_labels=3,
                ignore_mismatched_sizes=True
            ).to(self.device)
            self.model.eval()

    def train_on_cases(
        self,
        train_cases: List[Case],
        val_cases: Optional[List[Case]] = None,
        epochs: int = 5,
        batch_size: int = 16,
        lr: float = 2e-5,
        perspective_mode: str = "P4",
        save_dir: Optional[Path] = None
    ) -> Dict[str, Any]:
        """
        Fine-tune cross-encoder with graded loss.
        """
        self.load_model()
        from archehr_pipeline.data_loader import cases_to_sentence_dataframe
        train_df = cases_to_sentence_dataframe(train_cases)
        train_records = train_df.to_dict("records") if hasattr(train_df, "to_dict") else train_df

        train_ds = GradedSentenceDataset(train_records, self.tokenizer, perspective_mode=perspective_mode)
        train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=self.config.ranker_weight_decay)
        # Loss function with mild upweighting on essential and supplementary classes
        class_weights = torch.tensor([1.0, 2.0, 3.0], device=self.device)
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        self.model.train()
        history = []
        for epoch in range(epochs):
            total_loss = 0.0
            for batch in train_loader:
                optimizer.zero_grad()
                labels = batch.pop("labels").to(self.device)
                inputs = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**inputs)
                loss = criterion(outputs.logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                total_loss += loss.item()

            avg_loss = total_loss / max(1, len(train_loader))
            history.append({"epoch": epoch + 1, "train_loss": avg_loss})

        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
            self.model.save_pretrained(save_dir)
            self.tokenizer.save_pretrained(save_dir)

        self.model.eval()
        return {"history": history}

    @torch.no_grad() if _HAS_TORCH_HF else lambda fn: fn
    def score_sentences(
        self,
        case: Case,
        perspective_mode: str = "P4",
        batch_size: int = 16
    ) -> List[Dict[str, Any]]:
        """
        Score all sentences in the clinical note excerpt.
        Returns list of sentences with predicted relevance score:
          score = p(essential) * 2.0 + p(supplementary) * 1.0
        """
        if not case.sentences:
            return []

        if not _HAS_TORCH_HF or self.model is None:
            # Fallback heuristic: word overlap / BM25-style lexical matching for testing without PyTorch
            q_words = set(re.findall(r"\w+", (case.clinician_question + " " + case.patient_question).lower()))
            scored = []
            for s in case.sentences:
                s_words = set(re.findall(r"\w+", s["text"].lower()))
                overlap = len(q_words & s_words) / max(1, len(s_words))
                scored.append({
                    "id": s["id"],
                    "text": s["text"],
                    "score": float(overlap),
                    "p_essential": float(overlap),
                    "p_supplementary": 0.0
                })
            return sorted(scored, key=lambda x: x["score"], reverse=True)

        self.load_model()
        scored_sentences = []

        for start in range(0, len(case.sentences), batch_size):
            chunk = case.sentences[start:start + batch_size]
            pairs = [
                format_cross_encoder_input(
                    clinician_q=case.clinician_question,
                    patient_q=case.patient_question,
                    narrative=case.patient_narrative,
                    sentence_text=s["text"],
                    perspective_mode=perspective_mode
                )
                for s in chunk
            ]
            qa_list = [p[0] for p in pairs]
            qb_list = [p[1] for p in pairs]

            inputs = self.tokenizer(
                qa_list,
                qb_list,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)

            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

            for s, p in zip(chunk, probs):
                # Graded expected value: 2*p(essential) + 1*p(supplementary)
                relevance_score = float(2.0 * p[2] + 1.0 * p[1])
                scored_sentences.append({
                    "id": s["id"],
                    "text": s["text"],
                    "score": relevance_score,
                    "p_essential": float(p[2]),
                    "p_supplementary": float(p[1]),
                    "p_not_relevant": float(p[0])
                })

        return sorted(scored_sentences, key=lambda x: x["score"], reverse=True)

    def select_top_k_context(
        self,
        case: Case,
        k: Optional[int] = None,
        perspective_mode: str = "P4"
    ) -> List[Dict[str, Any]]:
        """
        Select the top K sentences from the clinical note excerpt.
        """
        k = k or self.config.k_context
        scored = self.score_sentences(case, perspective_mode=perspective_mode)
        return scored[:min(k, len(scored))]
